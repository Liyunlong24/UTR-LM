# coding: utf-8
import argparse
import logging
import os
import sys
import torch
import pytorch_lightning as pl
import deepspeed
import debugger

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from config import UTRLMConfig,model_config
from data.alphabet import Alphabet
from data.pretrained.datamodule import PretrainedDataModule
from model.model import RiNALMo
from utils.loss import UTRLMLoss
from pytorch_lightning import seed_everything
from utils.tensor_utils import tensor_tree_map
from utils.argparse_utils import remove_arguments
from utils.exponential_moving_average import ExponentialMovingAverage
from utils.lr_schedulers import AlphaFoldLRScheduler2
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from utils.callbacks import (EarlyStoppingVerbose,)
from utils.zero_to_fp32 import (get_fp32_state_dict_from_zero_checkpoint,get_global_step_from_zero_checkpoint)

class UTRMLMWrapper(pl.LightningModule):
    def __init__(
            self, config, tokenizer,lm_config: str = "nano",
    ):
        super(UTRMLMWrapper, self).__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.model = RiNALMo(model_config(lm_config))#只有在这用到了config.py中的138行前的一些参数配置

        for name, param in self.model.named_parameters():
            if "contact_head" in name:
                param.requires_grad = False

        self.loss = UTRLMLoss(config.loss)
        #找到了
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay #decay=0.999
        )
        self.cached_weights = None
        self.last_lr_step = -1

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}",
                indiv_loss,
                on_step=train, on_epoch=(not train), logger=True,
                batch_size=self.config.data.batch_size,
            )

            if (train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True,
                    batch_size=self.config.data.batch_size,
                )

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch,
                outputs,
            )

        for k, v in other_metrics.items():
            self.log(
                f"{phase}/{k}",
                torch.mean(v),
                on_step=False, on_epoch=True, logger=True,
                batch_size=self.config.data.batch_size,
            )

    def training_step(self, batch, batch_idx):
        if (self.ema.device != batch["tokens"].device):
            self.ema.to(batch["tokens"].device)

        outputs = self.model(batch["tokens"])
        loss, loss_breakdown = self.loss(outputs, batch, _return_breakdown=True)
        self._log(loss_breakdown, batch, outputs, train=True)
        # self.find_unused_parameters()
        #print('一轮训练结束')
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        #print('开始一轮验证')
        if self.config.globals.use_ema and self.cached_weights is None:
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])

        outputs = self.model(batch["tokens"])

        loss, loss_breakdown = self.loss(outputs, batch, _return_breakdown=True)
        self._log(loss_breakdown, batch, outputs, train=False)

    def on_validation_epoch_end(self):
        # Restore the model weights to normal
        if self.config.globals.use_ema:
            self.model.load_state_dict(self.cached_weights)
            self.cached_weights = None

    def _compute_validation_metrics(self,batch,outputs, ):
        metrics = {}

        # secstr acc
        if "label_secstr" in batch and "logits_ss" in outputs:
            secstr_acc = (outputs["logits_ss"].argmax(-1) == batch["label_secstr"]).float()
            secstr_mask = batch["label_secstr"] != -1  # (B, N)
            secstr_acc = (secstr_acc * secstr_mask).sum(-1) / (secstr_mask.sum(-1) + 1e-5)

            metrics["secstr_acc"] = secstr_acc
        return metrics

    def find_unused_parameters(self) -> None:
        print("on_after_backward enter")
        for name, p in self.model.named_parameters():
            if p.grad is None:
                print(name)
        print("on_after_backward exit")

    def configure_optimizers(self) -> torch.optim.Adam:

        optim_config = self.config.optimizer#config.py中223行代码体现的
        scheduler_config = self.config.scheduler

        param_optimizer = filter(lambda p: p.requires_grad, self.model.parameters())

        # Ignored as long as a DeepSpeed optimizer is configured
        #if isinstance(self.trainer.training_type_plugin, DeepSpeedStrategy):
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            if "offload_optimizer" in self.trainer.training_type_plugin.config['zero_optimization']:
                logging.info("cpu_offload enabled! Use deepspeed cpu adam!")
                optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
                    param_optimizer,
                    lr=optim_config.lr,
                    betas=(0.9, 0.999),
                    weight_decay=optim_config.weight_decay,
                    eps=optim_config.eps,
                )
            else:
                logging.info("cpu_offload disabled! Use deepspeed fused adam!")
                optimizer = deepspeed.ops.adam.FusedAdam(
                    param_optimizer,
                    lr=optim_config.lr,
                    betas=(0.9, 0.999),
                    weight_decay=optim_config.weight_decay,
                    eps=optim_config.eps,
                )
        else:#执行的是这个，优化器
            optimizer = torch.optim.Adam(
                param_optimizer,
                lr=optim_config.lr,
                weight_decay=optim_config.weight_decay,
                eps=optim_config.eps,
            )

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = optim_config.lr

        #学习率调度器的选用
        if scheduler_config.type == "alphafold":
            lr_scheduler = AlphaFoldLRScheduler2(
                optimizer,
                **scheduler_config,
            )
            scheduler_dict = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "AlphaFoldLRScheduler2",
            }
        # cosine annealing
        elif scheduler_config.type == "cosine":
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.lr_cosine_length,
            )
            scheduler_dict = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "CosineAnnealingLR",
            }

        # reduce on plateau#本项目用的是这个学习率调度器
        elif scheduler_config.type == "plateau":
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_config.lr_factor,
                patience=scheduler_config.lr_patience,
                min_lr=scheduler_config.lr_min,
            )

            scheduler_dict = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
                "name": "ReduceLROnPlateau",
            }
        elif scheduler_config.type == "none":
            scheduler_dict = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_config.type}")

        if scheduler_dict is not None:
            return [optimizer], [scheduler_dict]
        else:
            return [optimizer]

    def on_load_checkpoint(self, checkpoint):
        if hasattr(self, "ema"):
            ema = checkpoint["ema"]
            self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "ema"):
            checkpoint["ema"] = self.ema.state_dict()

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step

    def load_from_zero_checkpoint(self, zero_checkpoint_path):
        if not os.path.isfile(zero_checkpoint_path):
            raise FileNotFoundError(f"{zero_checkpoint_path}")
        sd = torch.load(zero_checkpoint_path, map_location="cpu")
        if "ema" in sd:
            sd = sd["ema"]["params"]
        else:
            sd = sd["module"]
            sd = {
                k[len("module.model."):]: v for k, v in sd.items() if k.startswith("module.model.")
            }
        self.model.load_state_dict(sd)
        # copy the weights to the ema
        if hasattr(self, "ema"):
            self.ema.load_state_dict_from_model(self.model)

    def load_from_checkpoint(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"{checkpoint_path}")
        sd = torch.load(checkpoint_path, map_location="cpu")
        if "ema" in sd:
            sd = sd["ema"]["params"]
        else:
            sd = sd["state_dict"]
            sd = {
                k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")
            }
        self.model.load_state_dict(sd)
        # copy the weights to the ema
        if hasattr(self, "ema"):
            self.ema.load_state_dict_from_model(self.model)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

def main(args):

    #命令行参数1，seed
    if (args.seed is not None):
        seed_everything(args.seed)

    #命令行参数2，yaml_config
    config = UTRLMConfig(
        name="pretraining",
        yaml_config=args.yaml_config
    )
    #命令行参数3，use_ema
    config.globals.use_ema = args.use_ema

    # data module
    #tokenizer = Alphabet.from_architecture("dna_lm")
    tokenizer = Alphabet()

    # PretrainedDataModule需要参数 tokenizer、max_len、batch_size、num_workers、train_path、
    # valid_path、test_path、 batch_seed、train_epoch_len

    #用到的是
    data_module = PretrainedDataModule(
        tokenizer=tokenizer,
        batch_seed=args.seed,
        **config.data,
    )
    # datamodule操作1，但是本项目中也不需要下载数据之类的操作，并不体现作用
    #data_module.prepare_data()
    data_module.setup()

    # 需要参数config.loss、config.ema.decay
    model_module = UTRMLMWrapper(config, tokenizer)

    # 处理模型从checkpoint恢复训练的情况，包括恢复学习率步数和模型权重
    if (args.resume_from_ckpt):
        if (os.path.isdir(args.resume_from_ckpt)):  # 如果是个目录就调用函数从目录中获取最后的全局步数
            last_global_step = get_global_step_from_zero_checkpoint(args.resume_from_ckpt)
        else:
            sd = torch.load(args.resume_from_ckpt, map_location="cpu")
            last_global_step = int(sd['global_step'])
        # 恢复学习率步数
        model_module.resume_last_lr_step(last_global_step)
        logging.info("Successfully loaded last lr step...")


    # 如果同时提供了两个参数，则恢复模型权重，而不是恢复训练状态
    if (args.resume_from_ckpt and args.resume_model_weights_only):
        # 不支持检查点是个目录
        if os.path.isdir(args.resume_from_ckpt):
            raise NotImplementedError(
                "Directory checkpoints are not supported now."
            )
        # 从0检查点恢复模型权重
        if args.resume_directly_from_mp_model_state:
            model_module.load_from_zero_checkpoint(args.resume_from_ckpt)
        else:  # 从普通检查点文件恢复模型权重
            model_module.load_from_checkpoint(args.resume_from_ckpt)
        logging.info(f"Successfully loaded model weights from {args.resume_from_ckpt}...")

    callbacks = []

    #用到,保存模型的条件和保存时机
    if (args.checkpoint_every_epoch):
        if args.deepspeed_config_path is not None or not args.wandb:
            dirpath = None
        else:
            dirpath = os.path.join(
                args.output_dir,
                args.wandb_project,
                args.wandb_version,
                "checkpoints",
            )
        mc = ModelCheckpoint(
            filename="epoch{epoch:02d}-step{step}-loss={val/loss:.3f}",
            dirpath=dirpath,
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            every_n_epochs=1,
            save_last=False,
            save_top_k=args.save_top_k,
        )
        callbacks.append(mc)
    #没用到
    if (args.early_stopping):
        # get the first task name
        # task_name = config.globals.tasks[0]
        es = EarlyStoppingVerbose(
            monitor=f"val/loss",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode="min",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)
    #用到了
    if (args.log_lr):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    loggers = []
    # 设置wandb
    if (args.wandb):
        wdb_logger = WandbLogger(
            name=args.experiment_name,
            save_dir=args.output_dir,
            version=args.wandb_version,
            project=args.wandb_project,
            #id=args.wandb_id,
            offline=True,
        )
        loggers.append(wdb_logger)

    # 分布式训练用的是ddp
   # if (args.deepspeed_config_path is not None):
       # strategy = DeepSpeedStrategy(
     #       config=args.deepspeed_config_path,
      #  )
       # if (args.wandb):
         #   wdb_logger.experiment.save(args.deepspeed_config_path)
    #elif (args.gpus is not None and args.gpus > 1) or args.num_nodes > 1:
    strategy = DDPStrategy(find_unused_parameters=False)
   # else:
       # strategy = None

    # 记录训练过程中的一些参数
    if (args.wandb):
        freeze_path = f"{wdb_logger.experiment.dir}/package_versions.txt"
        os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
        wdb_logger.experiment.save(f"{freeze_path}")
        wdb_logger.experiment.save("utr_lm/config.py")
        wdb_logger.experiment.save(args.yaml_config)

    trainer = pl.Trainer(
        #args,
        default_root_dir=args.output_dir,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        #accelerator='gpu',
        #devices=args.devices,
        #max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        log_every_n_steps=6,
        precision=args.precision,
    )

    if (args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt

    trainer.fit(
        model_module,
        datamodule=data_module,
        #ckpt_path=ckpt_path,
    )

def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')

if __name__ == '__main__':
    print('开始运行')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lm_config", type=str, default="nano",
        help="Language model configuration"
    )
    parser.add_argument(
        "-c", "--yaml_config", type=str, required=True,
    )
    parser.add_argument(
        "-o", "--output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    parser.add_argument(
        "--seed", type=int, default=2024,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_every_epoch", action="store_true", default=False,
        help="""Whether to checkpoint at the end of every training epoch"""
    )
    parser.add_argument(
        "--early_stopping", action="store_true", default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--save_top_k", type=int, default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--use_ema", type=bool_type, default=True,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--resume_directly_from_mp_model_state", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None,
        help="ID of a previous run to be resumed"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project to which this run will belong"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="wandb username or team name to which runs are attributed"
    )
    parser.add_argument(
        "--wandb_version", type=str, default=None,
        help="Sets the version, mainly used to resume a previous run."
    )
    parser.add_argument(
        "--log_lr", action="store_true", default=False,
        help="Whether to log the actual learning rate"
    )
    parser.add_argument(
        "--precision", type=str, default='16-mixed',
        help="Double precision, full precision, 16bit mixed precision or bfloat16 mixed precision"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=-1,
        help=" Stop training once this number of epochs is reached"
    )

   # parser = pl.Trainer.add_argparse_args(parser)
    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
    )
    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(
        parser,
        [
            "--accelerator",
            "--resume_from_checkpoint",
            "--reload_dataloaders_every_epoch",
            "--reload_dataloaders_every_n_epochs",
        ]
    )
    args = parser.parse_args()

    if (args.seed is None and (
            (args.gpus is not None and args.gpus > 1) or (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    if (str(args.precision) == "16" and args.deepspeed_config_path is not None):
        raise ValueError("DeepSpeed and FP16 training are not compatible")

    if args.yaml_config is not None:
        if not os.path.exists(args.yaml_config):
            raise FileNotFoundError(f"{os.path.abspath(args.yaml_config)}")
        args.yaml_config_basename = os.path.splitext(os.path.basename(args.yaml_config))[0]

    # process wandb args
    if args.wandb:
        if args.wandb_version is not None and args.yaml_config is not None:
            args.wandb_version = f"{args.yaml_config_basename}-{args.wandb_version}"
        if args.experiment_name is None:
            args.experiment_name = args.wandb_version
        wandb_log_dir = os.path.join(args.output_dir, "wandb")
        if not os.path.exists(wandb_log_dir):
            logging.info(f"generating directory for wandb logging located at {wandb_log_dir}")
            os.makedirs(wandb_log_dir, exist_ok=True)

    # This re-applies the training-time filters at the beginning of every epoch
    args.reload_dataloaders_every_n_epochs = 1

    main(args)

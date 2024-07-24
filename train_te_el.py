import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import os
import pytorch_lightning as pl

from torch.optim.lr_scheduler import ReduceLROnPlateau,SequentialLR, ConstantLR
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from torchmetrics.regression import R2Score
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef

import argparse
from pathlib import Path

from utr.config import model_config
from utr.model.model import RiNALMo
from utr.data.alphabet import Alphabet
from utr.data.downstream.teel.datamodule import TeelDataModule
from utr.model.downstream import RibosomeLoadingPredictionHead
from utr.utils.scaler import StandardScaler
from utr.utils.finetune_callback import GradualUnfreezing


class TeelPredictionWrapper(pl.LightningModule):
    def __init__(
            self,
            lm_config: str = "nano",
            head_embed_dim: int = 32,
            head_num_blocks: int = 6,
            lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.scaler = StandardScaler()

        self.lm = RiNALMo(model_config(lm_config))

        #for param in self.lm.parameters():
          #  param.requires_grad = False

        self.pred_head = RibosomeLoadingPredictionHead(
            c_in=self.lm.config['model']['transformer'].embed_dim,
            embed_dim=head_embed_dim,
            num_blocks=head_num_blocks
        )

        self.loss = nn.MSELoss()
        self.r2_metric = R2Score()
        self.pearson = PearsonCorrCoef()
        self.spearman = SpearmanCorrCoef()
        self.lr = lr

        self.pad_idx = self.lm.config['model']['embedding'].padding_idx

    def load_pretrained_lm_weights(self, pretrained_weights_path):

        checkpoint = torch.load(pretrained_weights_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 进行键名的调整，将预训练权重中的键名适配到当前模型中
        adapted_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                adapted_state_dict[k[6:]] = v  # 去掉 'model.' 前缀
            else:
                adapted_state_dict[f'model.{k}'] = v  # 添加 'model.' 前缀

        if "threshold" in adapted_state_dict:
            adapted_state_dict.pop("threshold")
        self.lm.load_state_dict(adapted_state_dict, strict=True, assign=False)
        print('加载模型成功')


    def forward(self, tokens):
        x = self.lm(tokens)["representation"]

        # Nullify padding token representations
        pad_mask = tokens.eq(self.pad_idx)
        x[pad_mask, :] = 0.0

        pred = self.pred_head(x, pad_mask)
        return pred

    def fit_scaler(self, batch):
        _, rl = batch
        self.scaler.partial_fit(rl)

    def _common_step(self, batch, batch_idx, log_prefix: str):
        seq_encoded, rl_target = batch
        preds = self(seq_encoded)
        scaled_rl_target = self.scaler.transform(rl_target)

        #print(f'模型预测：{preds},label：{rl_target},归一化后label:{scaled_rl_target}')

        loss = self.loss(preds, scaled_rl_target)

        preds = self.scaler.inverse_transform(preds)

        mse = F.mse_loss(preds, rl_target)
        mae = F.l1_loss(preds, rl_target)

        self.r2_metric.update(preds, rl_target)

        self.pearson.update(preds, rl_target)
        self.spearman.update(preds, rl_target)

        log = {
            f'{log_prefix}/loss': loss,
            f'{log_prefix}/mse': mse,
            f'{log_prefix}/mae': mae,
        }
        self.log_dict(log, sync_dist=True)

        return loss

    def _eval_step(self, batch, batch_idx, log_prefix):
        return self._common_step(batch, batch_idx, log_prefix=log_prefix)

    def _on_eval_epoch_start(self):
        # Reset metric calculator
        self.r2_metric.reset()
        self.pearson.reset()
        self.spearman.reset()

    def _on_eval_epoch_end(self, log_prefix: str):
        # Log and reset metric calculator
        if not self.trainer.sanity_checking:
            self.log(f"{log_prefix}/r2", self.r2_metric.compute(), sync_dist=True)
            self.log(f"{log_prefix}/pearson", self.pearson.compute(), sync_dist=True)
            self.log(f"{log_prefix}/spearman", self.spearman.compute(), sync_dist=True)
            self.r2_metric.reset()
            self.pearson.reset()
            self.spearman.reset()
    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            self.fit_scaler(batch)

        return self._common_step(batch, batch_idx, log_prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, log_prefix="val")

    def on_validation_epoch_start(self):
        return self._on_eval_epoch_start()

    def on_validation_epoch_end(self):
        return self._on_eval_epoch_end("val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, log_prefix="test")

    def on_test_epoch_start(self):
        return self._on_eval_epoch_start()

    def on_test_epoch_end(self):
        return self._on_eval_epoch_end("test")

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=5000) # TODO: Currently hardcoded!

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
        '''
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        #scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                        #     total_iters=300)  # TODO: Currently hardcoded!

        # scheduler = ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=0.98,
        #     patience=5,
        #     min_lr=1e-5,
        # )
        # exponential_lr_scheduler = ExponentialLR(optimizer, gamma=0.9)  # 之后逐步减小学习率
        initial_lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=50*46)  #

        exponential_lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01,total_iters=150*46)
        scheduler = SequentialLR(optimizer, schedulers=[initial_lr_scheduler, exponential_lr_scheduler],
                                 milestones=[40])

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #         "monitor": "train/loss",
        #     }
        # }

        return optimizer
        '''
def main(args):
    if args.seed:
        pl.seed_everything(args.seed)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Model
    model = TeelPredictionWrapper(
        lm_config=args.lm_config,
        head_embed_dim=args.embed_dim,
        head_num_blocks=args.num_blocks,
        lr=args.lr,
    )

    if args.pretrained_rinalmo_weights:
        model.load_pretrained_lm_weights(args.pretrained_rinalmo_weights)

    if args.init_params:
        model.load_state_dict(torch.load(args.init_params))

    # Datamodule
    alphabet = Alphabet()
    datamodule = TeelDataModule(
        data_root=args.data_dir,
        #task_type=args.task_type,
        alphabet=alphabet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,

    )

    # Set up callbacks and loggers
    callbacks = []
    loggers = []

    if args.wandb:
        wandb_logger = WandbLogger(
            save_dir=args.output_dir,
            version=args.wandb_version,
            project=args.wandb_project,
            offline=True,
            **{"entity": args.wandb_entity}
        )
        loggers.append(wandb_logger)


    dirpath = os.path.join(args.output_dir, "checkpoints")

    if args.checkpoint_every_epoch:
        epoch_ckpt_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename='epoch{epoch:02d}-step{step}-loss={val/loss:.3f}',
            every_n_epochs=1,
            save_top_k=5,
            auto_insert_metric_name = False,
            monitor = "val/loss",
            mode = "min",
            save_last = False,
        )
        callbacks.append(epoch_ckpt_callback)

    if loggers:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # Training
    if args.ft_schedule:
        ft_callback = GradualUnfreezing(
            unfreeze_schedule_path=args.ft_schedule,
        )
        callbacks.append(ft_callback)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision,
        default_root_dir=args.output_dir,
        log_every_n_steps=args.log_every_n_steps,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=loggers,
        callbacks=callbacks,
    )

    if not args.test_only:
        trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory with all the training and evaluation data"
    )
    parser.add_argument(
        "--init_params", type=str, default=None,
        help="""
        Path to the '.pt' file containing model weights that will be used
        as the starting point for the training (or evaluation)
        """
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for all the output files (checkpoints, logs, temporary files, etc.)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--checkpoint_every_epoch", action="store_true", default=False,
        help="Whether to checkpoint at the end of every training epoch"
    )
    parser.add_argument(
        "--test_only", action="store_true", default=False,
        help="""
        Skip the training and only run the evaluation on the test set
        (make sure to set '--ckpt_path' if you are using this option)
        """
    )

    # Model
    parser.add_argument(
        "--lm_config", type=str, default="nano",
        help="Language model configuration"
    )
    parser.add_argument(
        "--pretrained_rinalmo_weights", type=str, default=None,
        help="Path to the pretrained RiNALMo model weights"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=32,
        help="Prediction head embedding dimensionality"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=6,
        help="Number of transformer blocks in prediction head"
    )

    # W&B
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project to which this run will belong"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="Wandb username or team name to which runs are attributed"
    )
    parser.add_argument(
        "--wandb_version", type=str, default=None,
        help="Sets the version, mainly used to resume a previous run."
    )
    parser.add_argument(
        "--log_every_n_steps", type=int, default=50,
        help="How often to log within steps"
    )

    # Data
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="How many samples per batch to load"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="How many subprocesses to use for data loading"
    )
    parser.add_argument(
        "--pin_memory", action="store_true", default=False,
        help=" If activated, the data loader will copy Tensors into device/CUDA pinned memory before returning them"
    )

    # Training
    parser.add_argument(
        "--ft_schedule", type=str, default=None,
        help="Path to the fine-tuning schedule file"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--accelerator", type=str, default='auto',
        help="Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps”, “auto”)"
    )
    parser.add_argument(
        "--devices", type=str, default='auto',
        help="The devices to use for training"
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="Stop training after this number of steps"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=-1,
        help=" Stop training once this number of epochs is reached"
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=None,
        help="The value at which to clip gradients"
    )
    parser.add_argument(
        "--precision", type=str, default='16-mixed',
        help="Double precision, full precision, 16bit mixed precision or bfloat16 mixed precision"
    )
    parser.add_argument(
        "--task_type", type=str, required=True,
        help="You must specify whether it is a TE or EL task, which will process different data in the same file."#需要修改
    )

    args = parser.parse_args()
    main(args)

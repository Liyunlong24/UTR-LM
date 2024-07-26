# -*- coding: utf-8 -*-
export PYTHONPATH=/home/yunlongli/code/UTR-LM:$PYTHONPATH

mkdir -p '/home/yunlongli/code/UTR-LM/output/pretrain_300/logs'
ulimit -n 4096 && \
CUDA_VISIBLE_DEVICES="2,3" python3 \
    utr/train.py \
    --yaml_config ./ft_schedules/pretrain.yaml\
    --output_dir ./output/pretrain_300 \
    --log_lr \
    --max_epochs 300 \
    --wandb \
    --wandb_version v1 \
    --wandb_project pretrain_cosilsne_300 \
    --experiment_name pretrain_cosine_300 \
    --seed 2024 \
    --checkpoint_every_epoch \
    --save_top_k 20 \
    --use_ema true \
    >> ./output/pretrain_300/logs/train_exp2.log 2>&1
    #--replace_sampler_ddp=True \
    #--wandb_entity lyl \
    #--gpus 2 \
    #--precision 32 \
    # --log_every_n_steps 8 \
    # >> $output_dir/logs/train_exp2.log 2>&1

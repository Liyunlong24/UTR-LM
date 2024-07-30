# -*- coding: utf-8 -*-
export PYTHONPATH=/home/yunlongli/code/UTR-LM:$PYTHONPATH


mkdir -p '/home/yunlongli/code/UTR-LM/output/pretrain_plateau_300_1e_5_0.95_3_1e_7/logs'
ulimit -n 4096 && \
CUDA_VISIBLE_DEVICES="2,3" python3 \
    utr/train.py \
    --yaml_config ./ft_schedules/pretrain.yaml\
    --output_dir ./output/pretrain_plateau_300_1e_5_0.95_3_1e_7 \
    --log_lr \
    --max_epochs 500 \
    --wandb \
    --wandb_version pretrain_plateau_300_1e_5_0.95_3_1e_7 \
    --wandb_project pretrain \
    --seed 2024 \
    --checkpoint_every_epoch \
    --save_top_k 10 \
    --use_ema true \
    >> ./output/pretrain_plateau_300_1e_5_0.95_3_1e_7/logs/train_exp2.log 2>&1


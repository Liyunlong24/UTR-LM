# -*- coding: utf-8 -*-
export PYTHONPATH=/home/yunlongli/code/utr:$PYTHONPATH

mkdir -p '/home/yunlongli/code/RiNALMo/output/pretrain_v1/logs'
ulimit -n 4096 && \
CUDA_VISIBLE_DEVICES="2,3" python3 \
    utr/train.py \
    --yaml_config ./ft_schedules/pretrain.yaml\
    --output_dir ./output/pretrain_v1 \
    --log_lr \
    --max_epochs 5 \
    --wandb \
    --wandb_version v1 \
    --wandb_project utr_lm \
    --experiment_name utr_lm \
    --seed 2024 \
    --checkpoint_every_epoch \
    --save_top_k 10 \
    --use_ema true \
    >> $output_dir/logs/train_exp2.log 2>&1
    #--replace_sampler_ddp=True \
    #--wandb_entity lyl \
    #--gpus 2 \
    #--precision 32 \
    # --log_every_n_steps 8 \
    # >> $output_dir/logs/train_exp2.log 2>&1

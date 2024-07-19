# -*- coding: utf-8 -*-
mkdir -p '/home/yunlongli/code/UTR-LM/output/hek/te/logs'

CUDA_VISIBLE_DEVICES="0,1,2,3" python3 \
    train_te_el.py \
    --task_type TE \
    --pretrained_rinalmo_weights ./output/pretrain_v1/utr_lm/pretrain-v1/checkpoints/epoch46-step11750-loss=1.199.ckpt \
    --output_dir ./output/hek/te/ \
    --accelerator gpu \
    --max_epochs 100 \
    --batch_size 8 \
    --num_workers 0 \
    --seed 2024 \
    --lr 1e-3 \
    --wandb \
    --wandb_version v1 \
    --checkpoint_every_epoch \
    --wandb_experiment_name HEK_te \
    --wandb_project HEK_te \
    --data_dir ./data/HEK_sequence.csv \
    >> ./output/hek/te/logs/train_exp2.log 2>&1
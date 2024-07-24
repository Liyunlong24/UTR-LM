# -*- coding: utf-8 -*-

output_file="compared_mrl"
mkdir -p "/home/yunlongli/code/UTR-LM/output/$output_file/logs"

CUDA_VISIBLE_DEVICES="2,3" python3 \
    train_te_el.py \
    --task_type TE \
    --pretrained_rinalmo_weights ./output/pretrain_v4/checkpoints/epoch194-step3120000-loss=1.171.ckpt \
    --output_dir ./output/$output_file/ \
    --accelerator gpu \
    --max_epochs 20 \
    --batch_size 256 \
    --num_workers 8 \
    --seed 2024 \
    --lr 1e-3 \
    --wandb \
    --wandb_version $output_file \
    --checkpoint_every_epoch \
    --wandb_project compared \
    --data_dir ./data/HEK_sequence.csv \
    >> ./output/$output_file/logs/train_exp2.log 2>&1
# -*- coding: utf-8 -*-
seed=2024

task_type="TE"
dataset="HEK"

wandb_project="${task_type}_${dataset}"
#task_initiallr_epoch_scheduler_factor_patience_minlr
output_file="${wandb_project}_2e_4_200_ReduceLROnPlateau_0.9_5_5e_6"

mkdir -p "/home/yunlongli/code/UTR-LM/output/$output_file/logs"

CUDA_VISIBLE_DEVICES="2,3" python3 \
    train_te_el.py \
    --task_type $task_type \
    --pretrained_rinalmo_weights ./output/pretrain_v4/checkpoints/epoch194-step3120000-loss=1.171.ckpt \
    --output_dir ./output/$output_file/ \
    --accelerator gpu \
    --max_epochs 200 \
    --lr 2e-4 \
    --batch_size 1024 \
    --num_workers 8 \
    --wandb \
    --wandb_version $output_file \
    --checkpoint_every_epoch \
    --wandb_project $wandb_project\
    --data_dir $dataset \
    >> ./output/$output_file/logs/train_exp2.log 2>&1

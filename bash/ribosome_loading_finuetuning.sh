# -*- coding: utf-8 -*-
seed=2024
mkdir -p '/home/yunlongli/code/UTR-LM/output/TE_1e-3/logs'

CUDA_VISIBLE_DEVICES="2,3" python3 \
    train_ribosome_loading.py \
    --pretrained_rinalmo_weights ./output/pretrain_v4/checkpoints/epoch194-step3120000-loss=1.171.ckpt \
    --output_dir ./output/TE_1e-3/ \
    --accelerator gpu \
    --max_epochs 150 \
    --lr 2e-4 \
    --batch_size 1024 \
    --num_workers 8 \
    --wandb \
    --wandb_version TE_2e_4_ReduceLROnPlateau_0.85 \
    --checkpoint_every_epoch \
    --wandb_project TE_1e-3 \
    --data_dir ./data \
    >> ./output/TE_1e-3/logs/train_exp2.log 2>&1
    #--yaml_config 'ft_schedules/pretrain.yaml'\
    #--test_only \
    #--init_params ./weights/rinalmo_giga_ss_archiveII-5s_ft.pt \
    #--data_dir ./ss_data \
    #--prepare_data \
# -*- coding: utf-8 -*-
seed=2024
mkdir -p '/home/yunlongli/code/utr/output/ribosome_loading/logs'
CUDA_VISIBLE_DEVICES="2,3" python3 \
    train_ribosome_loading.py \
    --pretrained_rinalmo_weights ./output/pretrain_v1/utr_lm/pretrain-v1/checkpoints/epoch04-step40000-loss=1.176.ckpt \
    --output_dir ./output/ribosome_loading/ \
    --accelerator gpu \
    --max_epochs 10 \
    --batch_size 256 \
    --num_workers 8 \
    --wandb \
    --wandb_version v2 \
    --checkpoint_every_epoch \
    --wandb_experiment_name ribosome_loading \
    --wandb_project ribosome_loading \
    --data_dir ./utr/data/downstream/ribosome_loading/ribosom_loading_data \
    >> ./output/ribosome_loading/logs/train_exp2.log 2>&1
    #--yaml_config 'ft_schedules/pretrain.yaml'\
    #--test_only \
    #--init_params ./weights/rinalmo_giga_ss_archiveII-5s_ft.pt \
    #--data_dir ./ss_data \
    #--prepare_data \
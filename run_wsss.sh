#!/bin/bash

source activate wsss

MODEL_FLAGS="--num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True " # --learn_sigma True --class_cond True
DIFFUSION_FLAGS="--diffusion_steps 4000 "
TRAIN_FLAGS="--lr 3e-4 --batch_size 2"

python scripts/image_p_seg.py \
--save_dir {where_you_save_classifier} \
--model_path {where_you_save_diffusion} \
$MODEL_FLAGS $DIFFUSION_FLAGS -f 0 --batch_size 1 --guided




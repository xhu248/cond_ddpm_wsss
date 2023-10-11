#!/bin/bash

source activate wsss

MODEL_FLAGS="--num_channels 128 --num_res_blocks 3 " # --learn_sigma True --class_cond True
DIFFUSION_FLAGS="--diffusion_steps 4000 "
TRAIN_FLAGS="--lr 3e-4 --batch_size 2"

python train_tumor_ddfm.py --save_dir runs/results/diff_brats_2d_large \
--src_dir ../DDFM_USS/DATA/BraTS_patch/flair --dataset brats \
$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS




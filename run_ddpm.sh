#!/bin/bash

#$ -M xhu7@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 4    # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N diff_brats_2d # Specify job name


module load conda
module load mpich  # necessary for import package mpi4py
source activate diffusion

MODEL_FLAGS="--num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True " #--learn_sigma True --class_cond True
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 3e-4 --batch_size 2"

python train_tumor_ddfm.py --save_dir runs/results/diff_brats_2d_large \
--src_dir ../DATA/BraTS_patch/flair --dataset brats \
$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS



### sampling #####
#CUDA_VISIBLE_DEVICES=2 python image_sample.py --save_dir ./runs/results/diff_chaos_2d_large \
#$MODEL_FLAGS $DIFFUSION_FLAGS

#CUDA_VISIBLE_DEVICES=2 python image_sample.py --save_dir ./runs/results/diff_cond2_chaos_2d_large \
#$MODEL_FLAGS $DIFFUSION_FLAGS --class_index 2


CUDA_VISIBLE_DEVICES=2 python abdomen_cam.py --src_dir ../../vision_transformer/DATASET/chaos --dataset chaos \
--save_dir runs/results/guided_diff_cls_2_chaos_2d  $MODEL_FLAGS $DIFFUSION_FLAGS --aug_smooth --eigen_smooth --model_type encoder
--class_index 2
#CUDA_VISIBLE_DEVICES=2 python infer_features.py --src_dir ../../vision_transformer/DATASET/chaos --dataset chaos \
#--save_dir ./runs/results/diff_cond_2_chaos_2d $MODEL_FLAGS $DIFFUSION_FLAGS

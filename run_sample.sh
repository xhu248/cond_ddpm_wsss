#!/bin/bash

source activate wsss

MODEL_FLAGS="--num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True" #--learn_sigma True --class_cond True
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"

#python classifier_train.py --save_dir runs/results/guided_diff_cls_4_chaos_2d \
#--src_dir ../../vision_transformer/DATASET/chaos $TRAIN_FLAGS $CLASSIFIER_FLAGS

#python train_2d_ddfm.py \
#  --save_dir runs/results/diff_cond2_chaos_2d_large \
#  --src_dir ../../vision_transformer/DATASET/chaos --dataset chaos --class_index 2 \
#  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

#MODEL_FLAGS="--num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True" #--learn_sigma True --class_cond True
#python image_p_sample.py --src_dir ../../vision_transformer/DATASET/chaos --dataset chaos \
#--save_dir runs/results/guided_diff_cls_2_chaos_2d \
#--model_path runs/results/diff_cond2_chaos_2d_large/ema_0.9999.pt \
#$MODEL_FLAGS $DIFFUSION_FLAGS --class_index 2 --guided

python abdomen_cam.py \
--save_dir runs/results/guided_diff_cls_2_chaos_2d \
--src_dir ../../vision_transformer/DATASET/chaos --dataset chaos --class_index 2 \
-f 0 --aug_smooth --eigen_smooth --num_channels 128 --num_res_blocks 3


#MODEL_FLAGS="--num_channels 128 --num_res_blocks 3 "
#CUDA_VISIBLE_DEVICES=1 python image_p_sample.py --src_dir ../../vision_transformer/DATASET/chaos --dataset chaos \
#--save_dir runs/results/guided_diff_cls_2_chaos_2d \
#--model_path runs/results/diff_chaos_2d_large/model.pt \
#$MODEL_FLAGS $DIFFUSION_FLAGS --class_index 2 --guided
"""
Train a diffusion model on images.
"""

import argparse
import os, pickle, glob
import random

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_tumor_util import TrainLoop
from utils.tumor_utils import get_loader
from datasets.tumors.dataset import ImageDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def set_loader(args):
    normal_path = glob.glob(os.path.join(args.src_dir, "training", "normal", "*.jpg"))
    tumor_path = glob.glob(os.path.join(args.src_dir, "training", "tumor", "*.jpg"))
    print("Num of Tumor Data:", len(tumor_path))
    print("Num of Normal Data:", len(normal_path))
    data = normal_path + tumor_path
    print("data length", len(data))
    total_cases = len(data)

    print("============== Model Setup ===============")

    train_index = list(range(total_cases))
    # random.shuffle(train_index)
    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(train_index[-500:])

    dataset = ImageDataset(data)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                              pin_memory=False, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                            pin_memory=True, drop_last=True, sampler=test_sampler)

    return train_loader


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        image_size=args.img_size,
        model_name=args.model_name,
        dataset=args.dataset,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    data_loader = set_loader(args)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        class_cond=args.class_cond,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model_name='unet',
        img_size=240,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', default=0, type=int, help='fold number')
    parser.add_argument("--src_dir", default='../../vision_transformer/DATASET/mmwhs', type=str,
                        help='source dir of dataset ')
    parser.add_argument("--data_folder", default='preprocessed', type=str,
                        help='where to store results and logger')
    parser.add_argument('--workers', default=8, type=int, help='number of workers in dataset loader')
    parser.add_argument("--distributed", action='store_true', help='if start distribution')
    parser.add_argument("--save_dir", default='./run/results/diff_tumor', type=str,
                        help='where to store results and logger')
    parser.add_argument("--dataset", default='brats', type=str,
                        help='choose dataset')

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

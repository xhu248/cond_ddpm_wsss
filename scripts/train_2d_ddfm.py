"""
Train a diffusion model on images.
"""

import argparse
import os, pickle
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_2d_util import TrainLoop
from utils.tumor_utils import get_loader
from datasets.two_dim.NumpyDataLoader import NumpyDataSet


def set_loader(args):
    with open(os.path.join(args.src_dir, "splits.pkl"), 'rb') as f:
        splits = pickle.load(f)

    tr_keys = splits[args.fold]['train'] + splits[args.fold]['val']
    data_folder = os.path.join(args.src_dir, args.data_folder)
    train_loader = NumpyDataSet(data_folder, target_size=args.img_size, batch_size=args.batch_size, keys=tr_keys,
                                do_reshuffle=True, mode='val')
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
        class_index=args.class_index,
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
        save_interval=100,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model_name='unet',
        img_size=128,
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
    parser.add_argument("--dataset", default='hippocampus', type=str,
                        help='choose dataset')
    parser.add_argument('--class_index', default=1, type=int, help='choose which organ to consider')

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

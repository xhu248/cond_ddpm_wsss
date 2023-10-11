"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import pickle
import glob
import random

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
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
    train_sampler = SubsetRandomSampler(train_index[1000:-1000])
    test_sampler = SubsetRandomSampler(train_index[0:1000] + train_index[-1000:])

    dataset = ImageDataset(data)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                              pin_memory=False, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                            pin_memory=True, drop_last=True, sampler=test_sampler)

    return train_loader, val_loader


def convert_labels(labels, index=1):
    # convert pixel_wise labels to classfication labels
    cond = {}
    # labels=labels.squeeze()
    labels[labels != index] = 0
    cls_label = th.zeros(labels.size(0))
    for i in range(labels.size(0)):
        if th.sum(labels[i]) > 0:
            cls_label[i] = index
    cond['y'] = cls_label.long()
    return cond


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        image_size=args.img_size,
        in_channel=3,
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    data_loader, val_data = set_loader(args)


    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def forward_backward_log(data, labels, prefix="train"):
        data = data.to(dist_util.dev())
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(data.shape[0], dist_util.dev())
            data = diffusion.q_sample(data, t)
        else:
            t = th.zeros(data.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
                split_microbatches(args.microbatch, data, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            # losses[f"{prefix}_acc@5"] = compute_top_k(
            #     logits, sub_labels, k=5, reduction="none"
            # )
            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(data))

    step = 0
    while step < args.iterations - resume_step:
        for batch_data in data_loader:
            batch, labels = batch_data[0].float(), batch_data[1].long()
            labels = labels.to(dist_util.dev()).squeeze()
            forward_backward_log(batch, labels)
            logger.logkv("step", step + resume_step)
            logger.logkv(
                "samples",
                (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
            )
            if args.anneal_lr:
                set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
            mp_trainer.optimize(opt)

            if not step % args.log_interval:
                logger.dumpkvs()

            step += 1

        if val_data is not None and not step % args.eval_interval:
            eval_model(val_data, model)

        if (
                step
                and dist.get_rank() == 0
                and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, 0)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, 0)
    dist.barrier()


def eval_model(data_loader, model):
    total_logits = []
    total_labels = []
    total_losses = []
    logger.log("Validation")
    with th.no_grad():
        with model.no_sync():
            model.eval()
            for batch_data in data_loader:
                data, labels = batch_data[0].float(), batch_data[1].long()
                labels = labels.to(dist_util.dev()).squeeze()
                total_labels.append(labels)

                data = data.to(dist_util.dev())
                t = th.zeros(data.shape[0], dtype=th.long, device=dist_util.dev())
                logits = model(data, t)
                total_logits.append(logits)

                loss = F.cross_entropy(logits, labels, reduction="none")
                total_losses.append(loss)

            logger.log("average loss:", th.mean(th.cat(total_losses)))

            logits = th.cat(total_logits, dim=0)
            labels = th.cat(total_labels, dim=0)
            acc1 = compute_top_k(
                logits, labels, k=1, reduction="mean"
            )

            logger.log("average acc:", acc1)
    model.train()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"classifier_model.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=50,
        img_size=224,
    )
    defaults.update(classifier_and_diffusion_defaults())
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

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

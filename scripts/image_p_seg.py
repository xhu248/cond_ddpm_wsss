"""
Train a diffusion model on images.
"""

import argparse
import os
import glob
import time
import torch
import torch.distributed as dist
import numpy as np
import torch.nn as nn
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.filters import threshold_otsu
from utils.metrics import compute_dice, compute_mIOU, compute_hd
from utils.postprocessing import remove_all_but_the_largest_connected_component
from datasets.tumors.dataset import ImageDataset, SegmentDataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader


def set_loader(args):
    normal_path = glob.glob(os.path.join(args.src_dir, "validate", "normal", "*.jpg"))
    tumor_path = glob.glob(os.path.join(args.src_dir, "validate", "tumor", "*.jpg"))
    print("Num of Tumor Data:", len(tumor_path))
    print("Num of Normal Data:", len(normal_path))
    if args.fold == 0:
        data = tumor_path[:500]
    elif args.fold == 1:
        data = tumor_path[500:1000]
    elif args.fold == 2:
        data = tumor_path[1000:1500]
    elif args.fold == 3:
        data = tumor_path[1500:2000]

    print("data length", len(data))
    total_cases = len(data)

    print("============== Model Setup ===============")

    test_index = list(range(total_cases))
    # random.shuffle(train_index)
    #test_sampler = SubsetRandomSampler(test_index)
    test_sampler = SequentialSampler(test_index)

    dataset = SegmentDataset(data, seg_path=os.path.join(args.src_dir, "validate", "seg"))
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                            pin_memory=True, drop_last=True, sampler=test_sampler)

    return test_loader

def load_parameters(model, ckpt_path):
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)


def cond_fn(x, t, y, classifier):
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0]

#implementation of CG-Diff
def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        image_size=args.img_size,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    model_path = os.path.join(args.save_dir, "model.pt")
    load_parameters(model, model_path)
    model.eval()

    alpha = 0.0
    # build a clone model and modify the conditional embedding
    # with torch.no_grad():
    #     model2, _ = create_model_and_diffusion(
    #         image_size=args.img_size,
    #         **args_to_dict(args, model_and_diffusion_defaults().keys())
    #     )
    #     model2.to(dist_util.dev())
    #     load_parameters(model, model_path)
    #     model2.label_emb.weight[2] = alpha*model2.label_emb.weight[2] + (1 - alpha) * model2.label_emb.weight[0]
    #     model2.eval()

    logger.log("creating data loader...")
    data_loader = set_loader(args)

    # iterate through all data
    imgs = []
    lb = []
    slices = []
    slices_2 = []
    noises = []
    count = 0

    for batch in data_loader:
        data = batch[0].float().to(dist_util.dev())
        labels = batch[1]

        cond = {'y': batch[1].long().to(dist_util.dev())}
        if torch.sum(cond['y']) == 0:
            continue
        imgs.append(data.cpu().numpy())
        lb.append(labels.numpy())

        t = 300
        t = torch.tensor([t]).to(data.device)
        noisy_x = diffusion.q_sample(data, t)

        with torch.no_grad():
            print(cond['y'])
            sample = torch.clone(noisy_x)
            sample_2 = torch.clone(noisy_x)
            for i in range(300):
                t = t - 1
                #t = torch.tensor([t]).to(data.device)
                out = diffusion.p_mean_variance(model, sample, t, model_kwargs=cond)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(noisy_x.shape) - 1)))
                )
                sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(sample)

                fake_cond = {}
                fake_cond["y"] = torch.tensor([0]).long().to(dist_util.dev())
                out = diffusion.p_mean_variance(model, sample_2, t, model_kwargs=fake_cond)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(noisy_x.shape) - 1)))
                )
                sample_2 = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(sample_2)

        noises.append(noisy_x.squeeze().cpu().numpy())
        slices.append(sample.squeeze().cpu().numpy())
        slices_2.append(sample_2.squeeze().cpu().numpy())

        if count > 0:
            break
        else:
            count += 1

    noises = np.asarray(noises)
    slices = np.asarray(slices)
    slices_2 = np.asarray(slices_2)
    imgs = np.asarray(imgs)
    lb = np.asarray(lb)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    join = os.path.join

    if dist.get_rank() == 0:
        out_path = join(args.save_dir, "p_sample_cond.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, imgs, lb, noises, slices, slices_2)

    dist.barrier()
    logger.log("sampling complete")

# our method CDM
def main_ddim():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        image_size=args.img_size,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    load_parameters(model, args.model_path)
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(
        image_size=args.img_size,
        **args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(os.path.join(args.save_dir, "model.pt"), map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    logger.log("creating data loader...")
    data_loader = set_loader(args)

    # iterate through all data
    total_dice = []
    total_iou = []
    total_hd = []
    total_dice_1 = []
    total_iou_1 = []
    total_hd_1 = []

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    for batch in data_loader:
        data = batch[0].float().to(dist_util.dev())
        labels = batch[1].numpy().squeeze(1)
        logger.log(batch[3])

        cond = {'y': batch[2].squeeze(-1).long().to(dist_util.dev())}
        if torch.sum(cond['y']) == 0:
            continue

        t = 400
        t = torch.tensor([t]).to(dist_util.dev())
        noisy_x = diffusion.q_sample(data, t)

        with torch.no_grad():
            # sample = torch.clone(noisy_x)
            sample_2 = torch.clone(noisy_x)
            for i in range(10):
                t = t - 1
                # t = torch.tensor([t]).to(data.device)
                # out = diffusion.ddim_sample(model_fn, sample, t, model_kwargs=cond, cond_fn=cond_fn if args.guided else None)
                # sample = out["sample"]

                fake_cond = {}
                fake_cond["y"] = torch.tensor([0]).long().to(dist_util.dev())
                out = diffusion.ddim_sample(model_fn, sample_2, t, model_kwargs=fake_cond, cond_fn=cond_fn if args.guided else None)
                sample_2 = out["sample"]

        labels[labels > 0] = 1
        difftot = abs(data - sample_2).sum(dim=1).cpu().numpy()
        th = threshold_otsu(difftot)
        difftot[difftot > th] = 1
        difftot[difftot < th] = 0
        dice = compute_dice(labels, difftot)
        logger.log(dice)
        total_dice.append(dice)
        total_hd.append(compute_hd(difftot, labels))
        total_iou.append(compute_mIOU(labels, difftot))

        # kee the largest connected
        pred = np.zeros(difftot.shape)
        for k in range(pred.shape[0]):
            pred[k], _, _ = remove_all_but_the_largest_connected_component(difftot[k], [1], 1.0)
        total_dice_1.append(compute_dice(labels, pred))
        total_hd_1.append(compute_hd(pred, labels))
        total_iou_1.append(compute_mIOU(labels, pred))

    dist.barrier()
    logger.log("test complete")
    logger.log("average dice is:", np.average(total_dice))
    logger.log("average IoU is:", np.average(total_iou))
    logger.log("average hd is:", np.average(total_hd))
    logger.log("kee the largest connected")
    logger.log("average dice is:", np.average(total_dice_1))
    logger.log("average IoU is:", np.average(total_iou_1))
    logger.log("average hd is:", np.average(total_hd_1))


# our method CG-CDM
def main_ddim_guided():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        image_size=args.img_size,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    load_parameters(model, args.model_path)
    model.eval()

    alpha = 0.95
    # build a clone model and modify the conditional embedding
    with torch.no_grad():
        model2, _ = create_model_and_diffusion(
            image_size=args.img_size,
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model2.to(dist_util.dev())
        load_parameters(model2, args.model_path)
        model2.label_emb.weight[0] = alpha*model.label_emb.weight[args.class_index] + (1 - alpha) * model.label_emb.weight[0]
        model2.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(
        image_size=args.img_size,
        **args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(os.path.join(args.save_dir, "model.pt"), map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    logger.log("creating data loader...")
    data_loader = set_loader(args)

    logger.log("extracting features...")

    total_dice = []
    total_iou = []
    total_hd = []
    total_dice_1 = []
    total_iou_1 = []
    total_hd_1 = []

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    for batch in data_loader:
        data = batch[0].float().to(dist_util.dev())
        labels = batch[1].numpy().squeeze(1)

        cond = {'y': batch[2].squeeze(-1).long().to(dist_util.dev())}

        if torch.sum(cond['y']) == 0:
            continue

        t = 400
        t = torch.tensor([t]*args.batch_size).to(data.device)
        noisy_x = diffusion.q_sample(data, t)
        diff = torch.zeros(noisy_x.shape)

        with torch.no_grad():
            print(cond['y'])
            sample = torch.clone(noisy_x)
            sample_2 = torch.clone(noisy_x)
            for i in range(10):
                t = t - 1
                # t = torch.tensor([t]).to(data.device)
                out = diffusion.ddim_sample(model, sample, t, model_kwargs=cond, cond_fn=cond_fn if args.guided else None)
                sample = out["sample"]

                fake_cond = {}
                fake_cond["y"] = torch.tensor([0]*args.batch_size).long().to(dist_util.dev())
                out = diffusion.ddim_sample(model2, sample_2, t, model_kwargs=fake_cond, cond_fn=cond_fn if args.guided else None)
                sample_2 = out["sample"]

                diff += (sample - sample_2).cpu()

        labels[labels > 0] = 1
        diff = diff.sum(1).numpy()
        diff = (diff - diff.min()) / (diff.max() - diff.min())
        pred = np.zeros(diff.shape)
        for k in range(pred.shape[0]):
            th = threshold_otsu(diff[k])
            tmp = diff[k]
            tmp[tmp > th] = 1
            tmp[tmp < th] = 0
            pred[k] = tmp
        dice = compute_dice(labels, pred)
        logger.log(dice)
        total_dice.append(dice)
        total_hd.append(compute_hd(pred, labels))
        total_iou.append(compute_mIOU(labels, pred))

        # kee the largest connected
        pred_rm = np.zeros(pred.shape)
        for k in range(pred.shape[0]):
            pred_rm[k], _, _ = remove_all_but_the_largest_connected_component(pred[k], [1], 1.0)
        total_dice_1.append(compute_dice(labels, pred_rm))
        total_hd_1.append(compute_hd(pred_rm, labels))
        total_iou_1.append(compute_mIOU(labels, pred_rm))

    dist.barrier()
    logger.log("test complete")
    logger.log("average dice is:", np.average(total_dice))
    logger.log("average IoU is:", np.average(total_iou))
    logger.log("average hd is:", np.average(total_hd))
    logger.log("kee the largest connected")
    logger.log("average dice is:", np.average(total_dice_1))
    logger.log("average IoU is:", np.average(total_iou_1))
    logger.log("average hd is:", np.average(total_hd_1))


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model_name='unet',
        img_size=224,
        model_path="",
        classifier_scale=10.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
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
    parser.add_argument('--guided', action='store_true', help='choose whether use the guide of a classifier')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')
    parser.add_argument("--model_type", default='encoder', type=str,
                        help='choose model backbone')

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main_ddim_guided()
    #visualize_gradient()

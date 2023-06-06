"""
Train a diffusion model on images.
"""

import argparse
import os
import pickle
import torch
import torch.distributed as dist
import numpy as np
import random
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

from datasets.two_dim.NumpyDataLoader import NumpyDataSet
torch.manual_seed(0)
random.seed(0)

def load_parameters(model, ckpt_path):
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)


def set_loader(args):
    with open(os.path.join(args.src_dir, "splits.pkl"), 'rb') as f:
        splits = pickle.load(f)

    tr_keys = splits[args.fold]['test']
    data_folder = os.path.join(args.src_dir, args.data_folder)
    train_loader = NumpyDataSet(data_folder, target_size=args.img_size, batch_size=1, keys=tr_keys,
                                do_reshuffle=True, mode='val')
    return train_loader


def convert_labels(labels, index=1):
    # convert pixel_wise labels to classfication labels
    cond = {}
    #labels[labels != index] = 0
    if index == 2:
        labels[labels < 2] = 0
        labels[labels > 3] = 0
    else:
        labels[labels != index] = 0
    cls_label = torch.zeros(labels.size(0))
    for i in range(labels.size(0)):
        if torch.sum(labels[i]) > 0:
            cls_label[i] = index
    cond['y'] = cls_label.long().to(dist_util.dev()
                                    )
    return cond


def construct_jacobian(y, x, retain_graph=False):
    x_grads = []
    for idx, y_element in enumerate(y.flatten()):
        if x.grad is not None:
            x.grad.zero_()
        # if specified set retain_graph=False on last iteration to clean up
        y_element.backward(retain_graph=retain_graph or idx < y.numel() - 1)
        x_grads.append(x.grad.clone()[1])
    return torch.stack(x_grads).reshape(*y.shape, *x.shape)


def cond_fn(x, t, y, classifier):
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0]


def visualize_gradient():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

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

    imgs = []
    lb = []
    count = 0
    grad_maps = []
    t = 1000

    for batch in data_loader:
        data = batch['data'][0].float().to(dist_util.dev())
        labels = batch['seg'][0]
        cond = convert_labels(labels, args.class_index)['y'].to(dist_util.dev())

        imgs.append(data.cpu().numpy())
        lb.append(labels.numpy())

        t = torch.zeros(data.shape[0], dtype=torch.long, device=dist_util.dev())
        # noisy_x = diffusion.q_sample(data, t)
        t = t.float() * (1000.0 / 4000)


        gradient_map = cond_fn(data, t, cond, classifier)
        grad_maps.append(gradient_map.cpu().numpy())

        if count > 8:
            break
        else:
            count += 1

    grad_maps = np.asarray(grad_maps)
    imgs = np.asarray(imgs)
    lb = np.asarray(lb)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    join = os.path.join
    np.save(join(save_dir, "grad_maps.npy"), grad_maps)
    np.save(join(save_dir, "imgs.npy"), imgs)
    np.save(join(save_dir, "labels.npy"), lb)


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

    logger.log("extracting features...")

    # iterate through all data
    imgs = []
    lb = []
    slices = []
    slices_2 = []
    noises = []
    count = 0

    for batch in data_loader:
        data = torch.Tensor(batch['data'])[0].float().to(dist_util.dev())
        labels = batch['seg'][0]

        cond = convert_labels(labels, args.class_index)
        if torch.sum(cond['y']) == 0:
            continue
        imgs.append(data.cpu().numpy())
        lb.append(labels.numpy())

        t = 1500
        t = torch.tensor([t]).to(data.device)
        noisy_x = diffusion.q_sample(data, t)

        with torch.no_grad():
            print(cond['y'])
            sample = torch.clone(noisy_x)
            sample_2 = torch.clone(noisy_x)
            for i in range(1500):
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

    logger.log("extracting features...")

    # iterate through all data
    imgs = []
    lb = []
    slices = []
    slices_2 = []
    noises = []
    count = 0

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
        data = torch.Tensor(batch['data'])[0].float().to(dist_util.dev())
        labels = batch['seg'][0]

        cond = convert_labels(labels, args.class_index)
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
                # t = torch.tensor([t]).to(data.device)
                out = diffusion.ddim_sample(model_fn, sample, t, model_kwargs=cond, cond_fn=cond_fn if args.guided else None)
                sample = out["sample"]

                fake_cond = {}
                fake_cond["y"] = torch.tensor([0]).long().to(dist_util.dev())
                out = diffusion.ddim_sample(model_fn, sample_2, t, model_kwargs=fake_cond, cond_fn=cond_fn if args.guided else None)
                sample_2 = out["sample"]

        noises.append(noisy_x.squeeze().cpu().numpy())
        slices.append(sample.squeeze().cpu().detach().numpy())
        slices_2.append(sample_2.squeeze().cpu().detach().numpy())

        if count > 10:
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
        if args.guided:
            out_path = join(args.save_dir, "p_sample_ddim.npz")
        else:
            out_path = join(args.save_dir, "p_sample_ddim_cond.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, imgs, lb, noises, slices, slices_2)

    dist.barrier()
    logger.log("sampling complete")


def main_ddim_embedding():
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

    # iterate through all data
    imgs = []
    lb = []
    slices = []
    slices_2 = []
    slice_diff = []
    noises = []
    count = 0

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    for batch in data_loader:
        data = torch.Tensor(batch['data'])[0].float().to(dist_util.dev())
        labels = batch['seg'][0]

        cond = convert_labels(labels, args.class_index)
        if torch.sum(cond['y']) == 0:
            continue
        imgs.append(data.cpu().numpy())
        lb.append(labels.numpy())

        t = 500
        t = torch.tensor([t]).to(data.device)
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
                fake_cond["y"] = torch.tensor([0]).long().to(dist_util.dev())
                out = diffusion.ddim_sample(model2, sample_2, t, model_kwargs=fake_cond, cond_fn=cond_fn if args.guided else None)
                sample_2 = out["sample"]

                diff += abs(sample - sample_2).cpu()

        noises.append(noisy_x.squeeze().cpu().numpy())
        slices.append(sample.squeeze().cpu().detach().numpy())
        slices_2.append(sample_2.squeeze().cpu().detach().numpy())
        slice_diff.append(diff.squeeze().detach().numpy())

        if count > 10:
            break
        else:
            count += 1

    noises = np.asarray(noises)
    slices = np.asarray(slices)
    slices_2 = np.asarray(slices_2)
    imgs = np.asarray(imgs)
    lb = np.asarray(lb)
    diff_arr = np.asarray(slice_diff)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    join = os.path.join

    if dist.get_rank() == 0:
        if args.guided:
            out_path = join(args.save_dir, "p_sample_ddim_eg.npz")
        else:
            out_path = join(args.save_dir, "p_sample_ddim_cond_eg.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, imgs, lb, noises, slices, slices_2, diff_arr)

    dist.barrier()
    logger.log("sampling complete")


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
        img_size=128,
        model_path="",
        classifier_scale=1.0,
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
    parser.add_argument('--class_index', default=2, type=int, help='choose which organ to consider')
    parser.add_argument('--guided', action='store_true', help='choose whether use the guide of a classifier')

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main_ddim_embedding()
    #visualize_gradient()

import os
import math
import numpy as np
import torch
import random
from einops import rearrange
from monai import transforms, data
from monai.data import load_decathlon_datalist


def get_augmentation(args):
    """
                transforms.RandFlipd(keys=["image", "label"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=2),
                # transforms.RandRotate90d(
                #    keys=["image", "label"],
                #    prob=0.5,
                #    max_k=3,
                # ),
    """
    aug_list = transforms.Compose(
        [
            transforms.RandGaussianNoised(keys="image", prob=0.5),
            transforms.RandGibbsNoised(keys="image", prob=0.5),
            transforms.RandScaleIntensityd(keys="image",
                                           factors=0.1,
                                           prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image",
                                           offsets=0.1,
                                           prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    return aug_list


def patchify_augmentation(args, batch):
    aug_batch = dict()
    img = batch['image']
    label = batch['label']
    batch_size = img.size()[0]
    patch_dim = img.size()[-1] // args.mask_patch_size
    images_patch = rearrange(img, 'b c (h p1) (w p2) (d p3) -> (b h w d) c p1 p2 p3 ', p1=args.mask_patch_size // 2,
                             p2=args.mask_patch_size, p3=args.mask_patch_size)
    labels_patch = rearrange(label, 'b c (h p1) (w p2) (d p3) -> (b h w d) c p1 p2 p3 ', p1=args.mask_patch_size // 2,
                             p2=args.mask_patch_size, p3=args.mask_patch_size)

    mask = np.ones(images_patch.size()[0])
    # add noise mask to patches
    if args.add_contrast_mask:
        num_patches = images_patch.shape[0] // (args.mask_scale ** 3)
        num_mask = int(args.mask_ratio * num_patches)

        mask_index = np.random.permutation(num_patches)[:num_mask]
        mask = np.zeros(num_patches, dtype=int)
        mask[mask_index] = 1
        mask = mask.reshape(batch_size, patch_dim // args.mask_scale, patch_dim // args.mask_scale, patch_dim // args.mask_scale)
        if args.mask_scale > 1:
            mask = mask.repeat(args.mask_scale, axis=1).repeat(args.mask_scale, axis=2).repeat(args.mask_scale, axis=3)
        mask = mask.reshape(-1)

        noise_mask = torch.normal(mean=torch.zeros(num_mask * args.mask_scale ** 3, 1, args.mask_patch_size // 2,
                                                   args.mask_patch_size, args.mask_patch_size),
                                  std=0.1*torch.ones(num_mask * args.mask_scale ** 3, 1, args.mask_patch_size // 2,
                                                   args.mask_patch_size, args.mask_patch_size))
        images_patch[mask == 1] = noise_mask

    aug_list = get_augmentation(args)
    aug_batch["image"] = images_patch
    aug_batch["label"] = labels_patch
    aug_batch = aug_list(aug_batch)

    # reshape back to original size
    feature_size = img.shape[2] // args.mask_patch_size
    aug_batch["image"] = rearrange(aug_batch["image"], '(b h w d) c p1 p2 p3 -> b c (h p1) (w p2) (d p3)',
                                   h=patch_dim, w=patch_dim, d=patch_dim)
    aug_batch["label"] = rearrange(aug_batch["label"], '(b h w d) c p1 p2 p3 -> b c (h p1) (w p2) (d p3)',
                                   h=patch_dim, w=patch_dim, d=patch_dim)

    return aug_batch, mask


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, **x):
        return [self. transform(**x), self.transform(**x)]


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[:(self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0,high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    data_dir = "../../vision_transformer/DATASET/nnFormer_raw/nnFormer_raw_data/Task02_Synapse"
    datalist_json = args.json_file

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.Transposed(keys=["image", "label"], indices=[2, 0, 1]),
            # transforms.Spacingd(keys=["image", "label"],
            #                     pixdim=(args.space_x, args.space_y, args.space_z),
            #                     mode=("bilinear", "nearest")),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"],
                                    axcodes="RAS"),

            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            # transforms.Transposed(keys=["image", "label"], indices=[2, 0, 1]),
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest")),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"],
                                    axcodes="RAS"),
            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            # transforms.RandRotate90d(
            #     keys=["image", "label"],
            #     prob=1.0,
            #     max_k=1,
            # ),
            # transforms.RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            #     pos=1,
            #     neg=1,
            #     num_samples=2,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "validation",
                                            base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(test_ds,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     sampler=test_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json,
                                           True,
                                           "training",
                                           base_dir=data_dir)

        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist,
                transform=train_transform,
                cache_num=24,
                cache_rate=1.0,
                num_workers=args.workers,
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(train_ds,
                                       batch_size=args.batch_size,
                                       shuffle=(train_sampler is None),
                                       num_workers=args.workers,
                                       sampler=train_sampler,
                                       pin_memory=True,
                                       persistent_workers=True)
        val_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "validation",
                                            base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(val_ds,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        loader = train_loader

    return loader

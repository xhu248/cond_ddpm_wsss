import numpy as np
from sklearn.metrics import confusion_matrix
from medpy import metric


def compute_dice(gt, gen):
    # print(gen.dtype, gt.dtype)
    gt = gt.astype(np.uint8)
    gen = gen.astype(np.uint8)
    # print(gt.shape, gen.shape)
    inse = np.logical_and(gt, gen).sum()
    dice = (2. * inse + 1e-5) / (np.sum(gt) + np.sum(gen) + 1e-5)
    return dice

def compute_mIOU(gt, gen):
    gt = gt.astype(np.uint8)
    gen = gen.astype(np.uint8)
    intersection = np.logical_and(gt, gen)
    # print(intersection)
    union = np.logical_or(gt, gen)
    iou_score = (np.sum(intersection) + 1e-5) / (np.sum(union) + 1e-5)
    return iou_score

def compute_precision(gt, gen):
    # print(gen.dtype, gt.dtype)
    gt = gt.astype(np.uint8).flatten()
    gen = gen.astype(np.uint8).flatten()
    # print(gt.shape, gen.shape)
    tn, fp, fn, tp = confusion_matrix(gt, gen, labels=[0,1]).ravel()
    if (tp+fp) == 0:
        return 0.0
    else:
        precision = tp / (tp+fp)
        return precision

def compute_recall(gt, gen):
    # print(gen.dtype, gt.dtype)
    gt = gt.astype(np.uint8).flatten()
    gen = gen.astype(np.uint8).flatten()
    # print(gt.shape, gen.shape)
    tn, fp, fn, tp = confusion_matrix(gt, gen, labels=[0,1]).ravel()
    if (tp+fn) == 0:
        return 0.0
    else:
        recall = tp / (tp+fn)
        return recall

def compute_seg_metrics(gt, gen):
    dice = compute_dice(gt, gen)
    iou = compute_mIOU(gt, gen)
    precision = compute_precision(gt, gen)
    recall = compute_recall(gt, gen)
    return dice, iou, precision, recall


def compute_hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0
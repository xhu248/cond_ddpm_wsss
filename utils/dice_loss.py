import torch
import numpy as np
from torch import nn


def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            input = input.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(int(ax))
    return input


def mean_tensor(input, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            input = input.mean(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.mean(int(ax))
    return input


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., apply_nonlin=None, batch_dice=False, do_bg=True, smooth_in_nom=True,
                 background_weight=1, rebalance_weights=None):
        """
        hahaa no documentation for you today
        :param smooth:
        :param apply_nonlin:
        :param batch_dice:
        :param do_bg:
        :param smooth_in_nom:
        :param background_weight:
        :param rebalance_weights:
        """
        super(SoftDiceLoss, self).__init__()
        if not do_bg:
            assert background_weight == 1, "if there is no bg, then set background weight to 1 you dummy"
        self.rebalance_weights = rebalance_weights
        self.background_weight = background_weight
        self.smooth_in_nom = smooth_in_nom
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.y_onehot = None
        if not smooth_in_nom:
            self.nom_smooth = 0
        else:
            self.nom_smooth = smooth

    def forward(self, x, y):
        with torch.no_grad():
            y = y.long()
        shp_x = x.shape
        shp_y = y.shape
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))
        # now x and y should have shape (B, C, X, Y(, Z))) and (B, 1, X, Y(, Z))), respectively
        y_max = torch.max(y)
        y_onehot = torch.zeros(shp_x)
        if x.device.type == "cuda":
            y_onehot = y_onehot.cuda(x.device.index)
        # this is really fancy
        y_onehot.scatter_(1, y, 1)
        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]
        if not self.batch_dice:
            if self.background_weight != 1 or (self.rebalance_weights is not None):
                raise NotImplementedError("nah son")
            l = soft_dice(x, y_onehot, self.smooth, self.smooth_in_nom)
        else:
            l = soft_dice_per_batch_2(x, y_onehot, self.smooth, self.smooth_in_nom,
                                      background_weight=self.background_weight,
                                      rebalance_weights=self.rebalance_weights)
        return l


def soft_dice_per_batch(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1):
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    weights = torch.ones(intersect.shape)
    weights[0] = background_weight
    if net_output.device.type == "cuda":
        weights = weights.cuda(net_output.device.index)
    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth)) * weights).mean()
    return result


def soft_dice_per_batch_2(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1, rebalance_weights=None):
    if rebalance_weights is not None and len(rebalance_weights) != gt.shape[1]:
        rebalance_weights = rebalance_weights[1:]  # this is the case when use_bg=False
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    net_output_sqaure = sum_tensor(net_output*net_output, axes, keepdim=False)
    gt_square = sum_tensor(gt*gt, axes, keepdim=False)
    #fn = sum_tensor((1 - net_output) * gt, axes, keepdim=False)
    # fp = sum_tensor(net_output * (1 - gt), axes, keepdim=False)
    weights = torch.ones(intersect.shape)
    weights[0] = background_weight
    if net_output.device.type == "cuda":
        weights = weights.cuda(net_output.device.index)
    if rebalance_weights is not None:
        rebalance_weights = torch.from_numpy(rebalance_weights).float()
        if net_output.device.type == "cuda":
            rebalance_weights = rebalance_weights.cuda(net_output.device.index)
        intersect = intersect * rebalance_weights
        # fn = fn * rebalance_weights
    result = (1 - (2*intersect + smooth_in_nom)/(net_output_sqaure + gt_square + smooth) * weights)
    result = result[result > 0]  # ensure that when there is no target class, the dice loss is not too large
    result = result.mean()
    return result


def soft_dice(net_output, gt, smooth=1., smooth_in_nom=1.):
    axes = tuple(range(2, len(net_output.size())))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth))).mean()  #TODO: Was ist weights and er Stelle?
    return result


class MultipleOutputLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs that should predict the same y
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors
        l = weights[0] * self.loss(x[0], y)
        for i in range(1, len(x)):
            l += weights[i] * self.loss(x[i], y)
        return l
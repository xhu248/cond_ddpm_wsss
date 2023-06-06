# import denseCRF
import os
import numpy as np
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient)
from utils import *
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
from scipy.ndimage import label


def gen_seg_mask(img, cam, img_name, result_path, output_hist=False):
    # threshold = 0.8
    # if threshold < 0.1:
    #     final_seg = np.zeros_like(cam)
    # else:
    # first_seg = np.where(cam>threshold, 1, 0)
    # cam = (cam - cam.min())/ (cam.max() - cam.min())
    post_cam = DCRF(img, cam)
    final_seg = np.argmax(post_cam, axis=0)
    # final_seg = cam > 0.5

    return final_seg


def morphGAC(img, first_seg):
    # r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gimage = inverse_gaussian_gradient(img)
    final_seg = morphological_geodesic_active_contour(gimage, 300,
                                                      init_level_set=first_seg,
                                                      smoothing=2, balloon=-1)

    return final_seg


def DCRF(img, first_seg):
    # img = np.asarray(img)
    # img = (img*255).astype(np.uint8)

    # first_seg = first_seg.astype(np.float32)
    # prob = np.repeat(first_seg[..., np.newaxis], 2, axis=2)
    # # prob = prob[:, :, :2]
    # prob[:, :, 0] = 1.0 - prob[:, :, 0]
    # w1    = 10.0  # weight of bilateral term
    # alpha = 10    # spatial std
    # beta  = 13    # rgb  std
    # w2    = 3.0   # weight of spatial term
    # gamma = 3     # spatial std
    # it    = 50   # iteration
    # param = (w1, alpha, beta, w2, gamma, it)
    # final_seg = denseCRF.densecrf(img, prob, param)

    img = np.asarray(img)
    #img = (img * 255).astype(np.uint8)

    first_seg = first_seg.astype(np.float32)
    prob = np.repeat(first_seg[np.newaxis, ...], 2, axis=0)
    # prob = prob[:, :, :2]
    prob[0, :, :] = 1.0 - prob[0, :, :]
    scale_factor = 1.0
    h, w = img.shape[:2]
    n_labels = 2
    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(prob)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    img = np.ascontiguousarray(img.astype('uint8'))
    d.addPairwiseBilateral(sxy=10 / scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(10)
    final_seg = np.array(Q).reshape((n_labels, h, w))
    # print(final_seg.shape)
    return final_seg


def DCRF_nonRGB(img, first_seg):
    """
    :param img: [H, W], 2D images
    :param first_seg: [n_labels, H, W]
    :return: final_seg: seg after DCRF
    """
    prob = np.repeat(first_seg[np.newaxis, ...], 2, axis=0)
    prob[0, :, :] = 1.0 - prob[0, :, :]
    U = unary_from_softmax(prob)

    img = np.asarray(img[:, :, None])
    #img = (img * 255).astype(np.uint8)
    pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=img, chdim=2)

    h, w = img.shape[:2]
    n_labels = prob.shape[0]
    d = dcrf.DenseCRF2D(h, w, n_labels)
    d.setUnaryEnergy(U)
    d.addPairwiseEnergy(pairwise_energy, compat=10)
    Q = d.inference(10)
    final_seg = np.array(Q).reshape((n_labels, h, w))

    return final_seg


def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size

# import denseCRF
import os
import numpy as np
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient)
from utils import *
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
from scipy.ndimage import label


def gen_seg_mask(img, cam, img_name, result_path, output_hist=False):
    # threshold = 0.8
    # if threshold < 0.1:
    #     final_seg = np.zeros_like(cam)
    # else:
    # first_seg = np.where(cam>threshold, 1, 0)
    # cam = (cam - cam.min())/ (cam.max() - cam.min())
    post_cam = DCRF(img, cam)
    final_seg = np.argmax(post_cam, axis=0)
    # final_seg = cam > 0.5

    return final_seg


def morphGAC(img, first_seg):
    # r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gimage = inverse_gaussian_gradient(img)
    final_seg = morphological_geodesic_active_contour(gimage, 300,
                                                      init_level_set=first_seg,
                                                      smoothing=2, balloon=-1)

    return final_seg


def DCRF(img, first_seg):
    # img = np.asarray(img)
    # img = (img*255).astype(np.uint8)

    # first_seg = first_seg.astype(np.float32)
    # prob = np.repeat(first_seg[..., np.newaxis], 2, axis=2)
    # # prob = prob[:, :, :2]
    # prob[:, :, 0] = 1.0 - prob[:, :, 0]
    # w1    = 10.0  # weight of bilateral term
    # alpha = 10    # spatial std
    # beta  = 13    # rgb  std
    # w2    = 3.0   # weight of spatial term
    # gamma = 3     # spatial std
    # it    = 50   # iteration
    # param = (w1, alpha, beta, w2, gamma, it)
    # final_seg = denseCRF.densecrf(img, prob, param)

    img = np.asarray(img)
    #img = (img * 255).astype(np.uint8)

    first_seg = first_seg.astype(np.float32)
    prob = np.repeat(first_seg[np.newaxis, ...], 2, axis=0)
    # prob = prob[:, :, :2]
    prob[0, :, :] = 1.0 - prob[0, :, :]
    scale_factor = 1.0
    h, w = img.shape[:2]
    n_labels = 2
    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(prob)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    img = np.ascontiguousarray(img.astype('uint8'))
    d.addPairwiseBilateral(sxy=10 / scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(10)
    final_seg = np.array(Q).reshape((n_labels, h, w))
    # print(final_seg.shape)
    return final_seg


def DCRF_nonRGB(img, first_seg):
    """
    :param img: [H, W], 2D images
    :param first_seg: [n_labels, H, W]
    :return: final_seg: seg after DCRF
    """
    prob = np.repeat(first_seg[np.newaxis, ...], 2, axis=0)
    prob[0, :, :] = 1.0 - prob[0, :, :]
    U = unary_from_softmax(prob)

    img = np.asarray(img[:, :, None])
    #img = (img * 255).astype(np.uint8)
    pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=img, chdim=2)

    h, w = img.shape[:2]
    n_labels = prob.shape[0]
    d = dcrf.DenseCRF2D(h, w, n_labels)
    d.setUnaryEnergy(U)
    d.addPairwiseEnergy(pairwise_energy, compat=10)
    Q = d.inference(10)
    final_seg = np.array(Q).reshape((n_labels, h, w))

    return final_seg


def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size


def remove_all_but_the_largest_n_connected_component(image: np.ndarray, for_which_classes: list, kept_objects: int,
                                                     volume_per_voxel: float, minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            max_n_size = []
            sizes_list = list(object_sizes.values())
            sizes_list.sort()
            for i in range(kept_objects):
                if i > len(sizes_list):
                    break
                max_n_size.append(sizes_list[-i])

            kept_size[c] = np.sum(max_n_size)

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] not in max_n_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size
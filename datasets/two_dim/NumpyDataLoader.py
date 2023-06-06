import os
import fnmatch
import random

import numpy as np

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from datasets.data_loader import MultiThreadedDataLoader
from .data_augmentation import get_transforms

# get three parameters file (directory of processed images), files_len, slcies_ax( list of tuples)
def load_dataset(base_dir, pattern='*.npy', slice_offset=5, keys=None):
    fls = []
    files_len = []
    slices_ax = []

    for root, dirs, files in os.walk(base_dir):
        i = 0
        for filename in sorted(fnmatch.filter(files, pattern)):

            if keys is not None and filename in keys:
                npy_file = os.path.join(root, filename)
                numpy_array = np.load(npy_file, mmap_mode="r+")  # change "r" to "r+"

                fls.append(npy_file)
                files_len.append(numpy_array.shape[0]) # changed from 0 to 1

                slices_ax.extend([(i, j) for j in range(slice_offset, files_len[-1] - slice_offset)])

                i += 1
    return fls, files_len, slices_ax,


class NumpyDataSet(object):
    """
    TODO
    """
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000, seed=None, num_processes=8, num_cached_per_queue=8 * 4, target_size=128,
                 file_pattern='*.npy', label_slice=1, input_slice=(0,), do_reshuffle=True, keys=None):

        data_loader = NumpyDataLoader(base_dir=base_dir, mode=mode, batch_size=batch_size, num_batches=num_batches, seed=seed, file_pattern=file_pattern,
                                      input_slice=input_slice, label_slice=label_slice, keys=keys)

        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.number_of_slices = 1

        self.transforms = get_transforms(mode=mode, target_size=target_size)
        self.augmenter = MultiThreadedDataLoader(data_loader, self.transforms, num_processes=num_processes,
                                                 num_cached_per_queue=num_cached_per_queue, seeds=seed,
                                                 shuffle=do_reshuffle)
        self.augmenter.restart()

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        if self.do_reshuffle:
            self.data_loader.reshuffle()
        self.augmenter.renew()
        return self.augmenter

    def __next__(self):
        return next(self.augmenter)


class NumpyDataLoader(SlimDataLoaderBase):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 seed=None, file_pattern='*.npy', label_slice=1, input_slice=(0,), keys=None):

        self.files, self.file_len, self.slices = load_dataset(base_dir=base_dir, pattern=file_pattern, slice_offset=0, keys=keys, )
        super(NumpyDataLoader, self).__init__(self.slices, batch_size, num_batches)

        self.batch_size = batch_size

        self.use_next = False
        if mode == "train":
            self.use_next = False

        self.slice_idxs = list(range(0, len(self.slices)))  # divide 3D images into slices

        self.data_len = len(self.slices)

        self.num_batches = min((self.data_len // self.batch_size)+10, num_batches)

        if isinstance(label_slice, int):
            label_slice = (label_slice,)
        self.input_slice = input_slice
        self.label_slice = label_slice

        self.np_data = np.asarray(self.slices)

    def reshuffle(self):
        print("Reshuffle...")
        random.shuffle(self.slice_idxs)
        print("Initializing... this might take a while...")

    def generate_train_batch(self):
        open_arr = random.sample(self._data, self.batch_size)
        return self.get_data_from_array(open_arr)

    def __len__(self):
        n_items = min(self.data_len // self.batch_size, self.num_batches)
        return n_items

    def __getitem__(self, item):
        slice_idxs = self.slice_idxs
        data_len = len(self.slices)
        np_data = self.np_data


        if item > len(self):
            raise StopIteration()
        if (item * self.batch_size) == data_len:
            raise StopIteration()

        start_idx = (item * self.batch_size) % data_len
        stop_idx = ((item + 1) * self.batch_size) % data_len

        if ((item + 1) * self.batch_size) == data_len:
            stop_idx = data_len

        if stop_idx > start_idx:
            idxs = slice_idxs[start_idx:stop_idx]
        else:
            raise StopIteration()

        open_arr = np_data[idxs]  # tuple (a,b) of images of this batch

        return self.get_data_from_array(open_arr)

    def get_data_from_array(self, open_array):
        data = []
        fnames = []
        slice_idxs = []
        labels = []

        for slice in open_array:
            # slice is a tuple (a,b), slice[0] indicating which image it's,
            # and slice[1] incicats which one in the 3d image it's.
            fn_name = self.files[slice[0]]

            numpy_array = np.load(fn_name, mmap_mode="r")  # load data from .npy to numpy_arrary

            numpy_slice = numpy_array[slice[1]]  #(2,64,64)
            data.append(numpy_slice[None, self.input_slice[0]])   # 'None' keeps the dimension   (1,64,64)

            if self.label_slice is not None:
                labels.append(numpy_slice[None, self.label_slice[0]])   # 'None' keeps the dimension

            fnames.append(self.files[slice[0]])
            slice_idxs.append(slice[1])

        labels = np.asarray(labels)
        labels[labels > 7] = 0

        ret_dict = {'data': np.asarray(data), 'fnames': fnames, 'slice_idxs': slice_idxs}  # data_shape (8,1,64,64) 'data': np.asarray(data),
        if self.label_slice is not None:
            ret_dict['seg'] = labels

        return ret_dict




import os
import random
import shutil
import h5py

from PIL import Image
import numpy as np
from skimage.transform import resize
from sklearn.feature_extraction import image as skimg

import torch
from torch import nn
import torchvision

from transmission_model import TransmissionModel
from notebooks.residual_model import ResidualBlock, ResidualModel

class Model(nn.Module):
    """
    Model that combines CNN and RNN
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(img):
        """
        Forward pass of final model

        Input: img - input image
        Output: haze_free_image[0] - output filtered image
        """

        input_image_orig = np.asarray(Image.open(img)) / 255.0
        input_image = np.pad(input_image_orig, ((7, 8), (7, 8), (0, 0)), 'symmetric')

        model = TransmissionModel(input_image.shape)
        # model.load_weights('')

        input_image = np.expand_dims(input_image, axis=0)
        trans_map_orig = model.predict(input_image)
        trans_map = trans_map_orig.reshape(input_image_orig.shape[:2])
        trans_map_refine = TransmissionModel.transmission_refine((input_image_orig * 255.0).astype('uint8'), trans_map)

        res_map_input = input_image_orig / np.expand_dims(trans_map_refine, axis=(0, 3))

        model = ResidualModel(res_map_input.shape[1:])
        # model.load_weights('')
        res_map_output = model.predict(np.clip(res_map_input, 0, 1))

        haze_free_image = (res_map_input - res_map_output)
        haze_free_image = np.clip(haze_free_image, 0, 1)

        return haze_free_image[0]

    @staticmethod
    def load_train_dataset(count: int, patch_count: int):
        """
        Training dataset loading

        Input: count - count of images in one patch, patch_count - count of patches
        Output: clear_image_patch - patch of clear images, transmission_values - patch of transmission values, haze_image_patch - patch of hazy images
        """

        dataset = '/Users/daniilskrabo/Desktop/Practice STC/Athmoshere Filtering/data/nyu_depth_v2_labeled 2.mat'
        train_dataset = h5py.File(dataset, 'r')

        trans_vals = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

        nyu_image_patches = None
        nyu_haze_patches = None
        nyu_random_transmission = []

        for i in range(count):
            image = train_dataset['images'][i]
            image = (image.transpose(2, 1, 0)) / 255.0

            patches = skimg.extract_patches_2d(image, (16, 16), max_patches=patch_count)

            if nyu_image_patches is not None:
                nyu_image_patches = np.concatenate((nyu_image_patches, patches))
            else:
                nyu_image_patches = patches

            for patch in patches:
                transmission = random.choice(trans_vals)
                haze_patch = patch * transmission + (1 - transmission)
                nyu_random_transmission.append(transmission)

                if nyu_haze_patches is not None:
                    nyu_haze_patches = np.concatenate((nyu_haze_patches, [haze_patch]))
                else:
                    nyu_haze_patches = np.array([haze_patch])

        train_dataset.close()

        return {
            'clear_image_patch': nyu_image_patches,
            'transmission_values': nyu_random_transmission,
            'haze_image_patch': nyu_haze_patches
        }

    @staticmethod
    def create_train_dataset(count: int, patch_count: int, comp: int = 9, shuff: bool = True):
        """
        Training dataset creating

        Input: count - count of images in one patch, patch_count - count of patches, comp - data compression ratio, shuff - whether the data should be shuffled
        """

        d = Model.load_train_dataset(count=count, patch_count=patch_count)
        print('Dictionary created')

        with h5py.File('train_data.hdf5', 'w') as train_dataset:
            dset_clear_image = train_dataset.create_dataset('clear_image', data=d['clear_image_patch'],
                                                            compression='gzip', compression_opts=comp, shuffle=shuff)
            dset_trans_value = train_dataset.create_dataset('transmission_value', data=d['transmission_values'],
                                                            compression='gzip', compression_opts=comp, shuffle=shuff)
            dset_haze_image = train_dataset.create_dataset('haze_image', data=d['haze_image_patch'], compression='gzip',
                                                           compression_opts=comp, shuffle=shuff)

            print('compression:', dset_clear_image.compression, " compression_opt:", dset_clear_image.compression_opts,
                  " shuffle:", dset_clear_image.shuffle, "  size:", os.stat("train_data.hdf5").st_size)
            print('Dataset created')
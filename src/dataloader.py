import os, random, glob

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image

random.seed(1143)

def create_train_list(clear_images_path, hazy_images_path):

    train_list = []
    val_list = []

    image_list_hazy = glob.glob(hazy_images_path + '*.jpg')

    if not image_list_hazy:
        raise ValueError(f'No images found in {hazy_images_path}')
    
    tmp_dict = []

    for image in image_list_hazy:
        image = os.path.basename(image)
        key = image.split('_')[0] + '_' + image.split('_')[1] + '.jpg'

        if key in tmp_dict.keys():
            tmp_dict[key].append(image)
        else:
            tmp_dict[key] = []
            tmp_dict[key].append(image)

        train_keys = []
        val_keys = []

        len_keys = len(tmp_dict.keys())

        for i in range(len_keys):
            if i < len_keys * 9 / 10:
                train_keys.append(list(tmp_dict.keys())[i])
            else:
                val_keys.append(list(tmp_dict.keys())[i])
        
        for key in list(tmp_dict.keys()):
            if key in train_keys:
                for hazy_image in tmp_dict[key]:
                    train_list.append([os.path.join(clear_images_path, key), os.path.join(hazy_images_path, hazy_image)])
            else:
                for hazy_image in tmp_dict[key]:
                    val_list.append([os.path.join(clear_images_path, key), os.path.join(hazy_images_path, hazy_image)])

        random.shuffle(train_list)
        random.shuffle(val_list)

        return train_list, val_list
    
class DehazingLoader(data.Dataset):
    def __init__(self, clear_images_path, hazy_images_path, mode = 'train'):
        self.train_list, self.val_list = create_train_list(clear_images_path, hazy_images_path)

        if mode == 'train':
            self.data_list = self.train_list
            print('Total training examples: ', len(self.train_list))
        else:
            self.data_list = self.val_list
            print('Total validation examples: ', len(self.val_list))
        
        if len(self.data_list) == 0:
            raise ValueError(f'No data found for mode: {mode}')
        
        def __getitem__(self, index):
            data_clear_path, data_hazy_path = self.data_list[index]

            data_clear = Image.open(data_clear_path)
            data_hazy = Image.open(data_hazy_path)

            data_clear = data_clear.resize((480, 640), Image.Resampling.LANCZOS)
            data_hazy = data_hazy.resize((480, 640), Image.Resampling.LANCZOS)

            data_clear = (np.asarray(data_clear) / 255.0)
            data_hazy = (np.asarray(data_hazy) / 255.0)

            data_clear = torch.from_numpy(data_clear).float()
            data_hazy = torch.from_numpy(data_hazy).float()

            return data_clear.permute(2, 0, 1), data_hazy.permute(2, 0, 1)
        
        def __len__(self):
            return len(self.data_list)
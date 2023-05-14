import os
from torch.utils.data import Dataset
import torch
import random
import glob
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean


class RetinaDataset(Dataset):
    def __init__(self, data_folder, eval=False):
        self._data_folder = data_folder
        self._eval = eval
        self.build_dataset()

    def build_dataset(self):
        self._input_folder = self._data_folder + '/train/images' #os.path.join(self._data_folder, 'train/images')
        self._label_folder = self._data_folder + '/train/masques' # os.path.join(self._data_folder, 'train/masques')
        '''if self._eval:
            self._path_images = self._data_folder + '/val/images' # os.path.join(self._data_folder, 'eval')
            self._path_labels = self._data_folder + '/val/masques' #self._path_images'''


        if self._eval:
            self._input_folder = self._data_folder + '/mini_test' # os.path.join(self._data_folder, 'eval')
            self._label_folder = self._data_folder + '/val/masques' #self._path_images

        self._images = glob.glob(self._input_folder + "/*")
        # print(self._path_images)
        self._labels = glob.glob(self._label_folder + "/*")


        print(self._images)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = torch.from_numpy(img_as_ubyte(resize(io.imread(self._images[idx]), (512, 256))))
        label = torch.from_numpy(resize(io.imread(self._labels[idx]), (512, 256))).long()
        '''if random.randint(0, 1):
            image = image.flip(0)
            label = label.flip(0)
        if random.randint(0, 1):
            image = image.flip(1)
            label = label.flip(1)'''
        image = image.float().unsqueeze(0).unsqueeze(0)
        return image, label


class RetinaDatasetTest(Dataset):
    def __init__(self, data_folder, eval=False):
        self._data_folder = data_folder
        self._eval = eval
        self.build_dataset()

    def build_dataset(self):
        self._input_folder = self._data_folder + '/train/images' #os.path.join(self._data_folder, 'train/images')
        self._label_folder = self._data_folder + '/train/masques' # os.path.join(self._data_folder, 'train/masques')
        '''if self._eval:
            self._path_images = self._data_folder + '/val/images' # os.path.join(self._data_folder, 'eval')
            self._path_labels = self._data_folder + '/val/masques' #self._path_images'''


        if self._eval:
            self._input_folder = self._data_folder + '/mini_test' # os.path.join(self._data_folder, 'eval')
            self._label_folder = self._data_folder + '/val/masques' #self._path_images

        self._images = glob.glob(self._input_folder + "/*")
        # print(self._path_images)
        self._labels = glob.glob(self._label_folder + "/*")


        print(self._images)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = torch.from_numpy(img_as_ubyte(resize(io.imread(self._images[idx]), (512, 256))))
        label = self._images[idx]
        '''if random.randint(0, 1):
            image = image.flip(0)
            label = label.flip(0)
        if random.randint(0, 1):
            image = image.flip(1)
            label = label.flip(1)'''
        image = image.float().unsqueeze(0).unsqueeze(0)
        return image, label

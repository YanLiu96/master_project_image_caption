# 6.19 20:37
import os
import h5py
import json
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: saved_data/
        :param data_name:
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'train', 'val', 'test'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(data_folder+'hdf5_images/'+self.split+'_images.hdf5', 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(data_folder+'cap2vec/'+self.split+'_cap_vec.json', 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(data_folder+'cap2vec/'+self.split+'_cap_len.json', 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'train':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size

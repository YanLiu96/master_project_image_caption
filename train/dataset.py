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

        # Open hdf5 file where train/val/test images are stored
        self.h = h5py.File(data_folder+'hdf5_images/'+self.split+'_images.hdf5', 'r')
        self.imgs = self.h['images'] # get the images data
        self.cpi = self.h.attrs['captions_per_image'] # get the number of captions for one image (5)

        # Open caption (annotation) for images
        with open(data_folder+'cap2vec/'+self.split+'_cap_vec.json', 'r') as j:
            self.captions = json.load(j)

        # Get the number of words for each caption (annotation)
        with open(data_folder+'cap2vec/'+self.split+'_cap_len.json', 'r') as j:
            self.caplens = json.load(j)

        self.transform = transform # PyTorch transfoem instance

        # the size of the train/val/test dataset
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        """
        Get the i-th caption and the image (which is i/5 ) that corresponds to it
        """
        # turn the i/5-th image to tensor format
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)

        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i]) #turn the i-th caption to tensor format
        caplen = torch.LongTensor([self.caplens[i]]) #turn the i-th caption length to tensor format

        if self.split is 'train': # for trainning
            return img, caption, caplen
        else: # For validation and testing,also return all the 5 captions for the image to get the BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size

import os
import numpy as np
import h5py
import json

def images_save_in_hdf5():
    # for train subset
    for imgae_name, image_captions in zip(train_images_names, train_images_captions):
        create_hdf5_file()
    # for val subset
    for imgae_name, image_captions in zip(va_images_names, val_images_captions):
        create_hdf5_file()
    # for test subset
    for imgae_name, image_captions in zip(test_images_names, test_images_captions):
        create_hdf5_file() 

def create_hdf5_file():
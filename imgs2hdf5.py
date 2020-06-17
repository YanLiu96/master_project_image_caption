import os
import numpy as np
import h5py
import json
from progress.bar import Bar

#from data_preprocess import test_images_captions
def images_save_in_hdf5():
    # for train subset
    for imgae_name, image_captions in zip(train_images_names, train_images_captions):
        create_hdf5_file('train', imgae_name, image_captions,)
    # for val subset
    for imgae_name, image_captions in zip(va_images_names, val_images_captions):
        create_hdf5_file('val', imgae_name, image_captions)
    # for test subset
    for imgae_name, image_captions in zip(test_images_names, test_images_captions):
        create_hdf5_file('test', imgae_name, image_captions)

# http://docs.h5py.org/en/stable/quick.html
def create_hdf5_file(type, imgae_name, image_captions):
    with h5py.File(os.path.join('saved_data/hdf5_images', type + '_images.hdf5'), 'a') as f:
        images = f.create_dataset('images', (len(imgae_name), 3, 256, 256), dtype='uint8')
        print("%s images storage in progress\n" % type)
        bar = Bar('Processing', max=20)
        for path in impaths:
            bar.next() 
        bar.finish()
        
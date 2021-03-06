import os
import time
import h5py
import json
import numpy as np
import imageio

from PIL import Image
from progress.bar import ChargingBar

from prepare.data_preprocess import train_images_names, train_images_captions
from prepare.data_preprocess import val_images_names, val_images_captions
from prepare.data_preprocess import test_images_names, test_images_captions

def images_save_in_hdf5():
    # for train subset
    create_hdf5_file('train', train_images_names, train_images_captions,)
    # for val subset
    create_hdf5_file('val', val_images_names, val_images_captions)
    # for test subset
    create_hdf5_file('test', test_images_names, test_images_captions)

def create_hdf5_file(type, imgaes_names, image_captions):
    start = time.time()
    with h5py.File(os.path.join('saved_data/hdf5_images', type + '_images.hdf5'), 'a') as f:
        f.attrs['captions_per_image'] = 5
        images = f.create_dataset('images', (len(imgaes_names), 3, 256, 256), dtype='uint8')
        print("%s images storage in progress\n" % type)
        bar = ChargingBar('Processing '+type+' images in hdf5 file', max=len(imgaes_names))
        for i, path in enumerate(imgaes_names):
            img = imageio.imread(path)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            
            img=np.array(Image.fromarray(img).resize((256,256)))
            img = img.transpose(2, 0, 1)
            assert img.shape == (3, 256, 256)
            assert np.max(img) <= 255
            # Save image to HDF5 file
            images[i] = img
            bar.next() 
        bar.finish()
    print ('Time taken for saving {} images in hdf5 file: {} sec\n'.format(type, time.time() - start))
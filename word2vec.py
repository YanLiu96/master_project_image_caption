# Step4: word2vec.py created in 2020.6.18 20:44
# This file map the word to vector, in order to make the captions become vector
import os
import time
import json

from data_preprocess import train_images_names, train_images_captions
from data_preprocess import val_images_names, val_images_captions
from data_preprocess import test_images_names, test_images_captions
from data_preprocess import word2idx

# save caption vectors
def save_cap_vec():
    cap2vec('train',train_images_captions)
    cap2vec('val',val_images_captions)
    cap2vec('test',test_images_captions)

# save caption vectors
def cap2vec(type, dataset):
    idx=0
    cap_vec=[]
    cap_len=[]
    for caption in dataset[idx]:
        for word in caption:
            # if find word in vocabulary, return corresponding index
            # otherwise return the index of unknown
            vec=word2idx.get(word, word2idx['<unk>'])
            cap_vec.append(vec)
        cap_len.append(len(caption)+2)
        idx=idx+1
    with open(os.path.join('saved_data/cap2vec',type+'cap_vec.json'),'w') as f1:
        json.dump(cap_vec, f1)
    with open(os.path.join('saved_data/cap2vec'),type+'cap_len.json') as f2:
        json.dump(cap_len, f2)
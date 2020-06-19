# Step4: word2vec.py created in 2020.6.18 20:44
# This file map the word to vector, in order to make the captions become vector
import os
import time
import json
from progress.bar import Bar
from data_preprocess import train_images_names, train_images_captions
from data_preprocess import val_images_names, val_images_captions
from data_preprocess import test_images_names, test_images_captions
from data_preprocess import word2idx

# save caption vectors
def save_cap_vec(max_cap_len):
    cap2vec('train',train_images_captions, max_cap_len)
    cap2vec('val',val_images_captions, max_cap_len)
    cap2vec('test',test_images_captions, max_cap_len)

# 后续可以使用torch自带的库来处理
# save caption vectors
def cap2vec(type, dataset, max_cap_len):
    cap_vec=[]
    cap_len=[]
    print('\n')
    bar = Bar('Processing caption vector', max=len(dataset)*5)
    start = time.time()
    for captions in dataset:
        for caption in captions: #[ [[c1],[c2],[c3]],[[b1],[b2],[b3]] ] caption: ['a','b','c']
            temp=[]
            for word in caption:
                # if find word in vocabulary, return corresponding index
                # otherwise return the index of unknown
                temp.append(word2idx.get(word, word2idx['<unk>']))
            
            temp=[word2idx['<start>']]+temp+[word2idx['<end>']]+[word2idx['<pad>']] * (max_cap_len - len(caption))
            cap_vec.append(temp) #[[0,1,2,3,4....],[0,1,2,3,4....],[0,1,2,3,4....]....]
            cap_len.append(len(caption)+2)
            bar.next()

    with open(os.path.join('saved_data/cap2vec',type+'_cap_vec.json'),'w') as f1:
            json.dump(cap_vec, f1)

    with open(os.path.join('saved_data/cap2vec',type+'_cap_len.json'), 'w') as f2:
            json.dump(cap_len, f2)
    print('\n')
    print ('Time taken for saving {} caption vectors in json file: {} sec\n'.format(type, time.time() - start))
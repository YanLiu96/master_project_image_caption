# Step2: data_preprocess.py created in 2020.6.16 08:19

# This file is used for preprocess downloaded data
# 1. splite dataset into train, valuate, test subset.
# 2. 
import os
import json
from collections import Counter
from random import seed, choice, sample

train_images_names=[]
train_images_captions=[]
val_images_names=[]
val_images_captions=[]
test_images_names=[]
test_images_captions=[]
word_frequency = Counter()
vocabulary=[]
word2idx={}


def split_dataset(caption_file, max_caption_len):
    with open(caption_file, 'r') as j:
        annotation = json.load(j)
    
    for image in annotation['images']:
        caps_for_image = []
        for caption in image['sentences']:
        #caption format is: 'sentences': [{'tokens': ['a', 'man', 'with', 'a', 'red', 'helmet', 'on', 'a', 'small', 'moped', 'on', 'a', 'dirt', 'road'], 'raw': 'A man with a red helmet on a small moped on a dirt road. ', 'imgid': 0, 'sentid': 770337}
            word_frequency.update(caption['tokens'])
            if len(caption['tokens']) <= max_caption_len:
                caps_for_image.append(caption['tokens'])

        if len(caps_for_image) == 0:
            continue
        # because the number of captions of  some images is not 5. So make up the difference
        caps_for_image = make_up_diff(caps_for_image)

        path = os.path.join("coco_dataset", image['filepath'], image['filename']) 
        if image['split'] in {'train', 'restval'}:
            train_images_names.append(path)
            train_images_captions.append(caps_for_image)
        elif image['split'] in {'val'}:
            val_images_names.append(path)
            val_images_captions.append(caps_for_image)
        elif image['split'] in {'test'}:
            test_images_names.append(path)
            test_images_captions.append(caps_for_image)
      
    try:
        len(train_images_names) == len(train_images_captions) # 113287
        len(val_images_names) == len(val_images_captions) # 5000
        len(test_images_names) == len(test_images_captions) # 5000
    except Exception as e:
        print(e)

    print('--Coco dataset has already been splited into train, val, test subsets--')
    print('--The length of training set is %s --' %len(train_images_names))


def make_up_diff(caps_for_image):
    if len(caps_for_image)<5:
        miss_quantity = 5-len(caps_for_image)
        seed(123)
        for i in range(miss_quantity):
            caps_for_image = caps_for_image + [choice(caps_for_image)]
    else:
        caps_for_image = sample(caps_for_image, k=5)
    assert len(caps_for_image) == 5
    return caps_for_image


def build_vocabulary_word2idx(min_word_freq):
    # Create word map
    for word in word_frequency.keys():
        if word_frequency[word] > min_word_freq:
            vocabulary.append(word)
    for v,k in enumerate(vocabulary):
        word2idx.update({k:v+1})
    word2idx['<unk>'] = len(word2idx) + 1
    word2idx['<start>'] = len(word2idx) + 1
    word2idx['<end>'] = len(word2idx) + 1
    word2idx['<pad>'] = 0
    print('The size of vocabulary is %s' %len(vocabulary))
    print('The size of word2idx is %s' %len(word2idx))

def save_word2idx(word2idx_save_path):
    with open(os.path.join(word2idx_save_path,'word2idx.json'), 'w') as j:
        json.dump(word2idx, j)

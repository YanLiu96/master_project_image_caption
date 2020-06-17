# Step2: data_preprocess.py created in 2020.6.16 08:19

# This file is used for preprocess downloaded data
# 1. splite dataset into train, valuate, test subset.
# 2. 
import os
import json
from collections import Counter
# def create_input_files(dataset, caption_file, image_folder, num_of_caption_per_image, min_word_freq, output_folder, max_len=100):
#     assert dataset in {'coco', 'flickr8k', 'flickr30k'}
#     # 
#     with open(karpathy_json_path, 'r') as j:
#         data = json.load(j)
train_images_names=[]
train_images_captions=[]
val_images_names=[]
val_images_captions=[]
test_images_names=[]
test_images_captions=[]
word_frequency = Counter()
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
    assert len(train_images_names) == len(train_images_captions) # 113287
    assert len(val_images_names) == len(val_images_captions) # 5000
    assert len(test_images_names) == len(test_images_captions) # 5000
    print('--coco dataset has already been splited into train, val, test subsets--')

split_dataset('coco_dataset/caption_datasets/dataset_coco.json', 50)

def build_vocabulary_word2idx(min_word_freq):
    # Create word map
    vocabulary=[]
    word2idx={}
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
build_vocabulary_word2idx(5)
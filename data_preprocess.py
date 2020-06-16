# Step2: data_preprocess.py created in 2020.6.16 08:19

# This file is used for preprocess necessary data.
# 1. coco train2014 dataset called train2014.zip and unzip it.
# 2. the annotations(captions) for train2014.
def create_input_files(dataset, caption_file, image_folder, num_of_caption_per_image, min_word_freq, output_folder, max_len=100):
    assert dataset in {'coco', 'flickr8k', 'flickr30k'}
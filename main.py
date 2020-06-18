from data_preprocess import split_dataset, build_vocabulary_word2idx, save_word2idx
from imgs2hdf5 import images_save_in_hdf5
if __name__ == '__main__':
    split_dataset('coco_dataset/caption_datasets/dataset_coco.json', 50)
    build_vocabulary_word2idx(5)
    save_word2idx('saved_data/word2index')
    images_save_in_hdf5()
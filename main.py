from prepare.data_preprocess import split_dataset, build_vocabulary_word2idx, save_word2idx
#from prepare import images_save_in_hdf5
from prepare.word2vec import save_cap_vec

if __name__ == '__main__':
    split_dataset('coco_dataset/caption_datasets/dataset_coco.json', 50)
    build_vocabulary_word2idx(5)
    save_word2idx('saved_data/word2index')
    #images_save_in_hdf5()
    save_cap_vec(50)
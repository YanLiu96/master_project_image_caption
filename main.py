import argparse
from prepare.data_preprocess import split_dataset, build_vocabulary_word2idx, save_word2idx
from prepare.imgs2hdf5 import images_save_in_hdf5
from prepare.word2vec import save_cap_vec
from train.train import main_train
from test.discrible import caption_img

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Yan Liu master project: let computer tell what the image is about')

    parser.add_argument('--control', '-c', help='control:data_prepare, train')
    parser.add_argument('--test', '-t', help='get the caption for the given image')
    # parser.add_argument('--train', '-tr', help='train the models')
    # parser.add_argument('--test', '-t', help='image path to be tested')
    args = parser.parse_args()
    if args.control == 'data_prepare':
        split_dataset('coco_dataset/caption_datasets/dataset_coco.json', 50)
        build_vocabulary_word2idx(5)
        save_word2idx('saved_data/word2index')
        #images_save_in_hdf5()
        save_cap_vec(50)
    elif args.control == 'train':
        main_train()
    elif args.test:
        caption_img(args.test)
    else:
        print('Choose the things you want to do, you can use -h to get help')

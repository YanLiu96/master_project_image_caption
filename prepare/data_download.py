import os

def download():
# # Download coco image files
# image_folder = '/coco_dataset/train2014/'
# if not os.path.exists(os.path.abspath('.') + image_folder):
#     os.system('wget http://images.cocodataset.org/zips/train2014.zip')
#     os.system('unzip train2014.zip')
#     os.remove('train2014.zip')
#     print("coco train2014 images have been downloaded")
# else:
#     print("coco train2014 images have already exist")

# Download coco annotation desgined by Andrej
    annotation_folder = '/coco_dataset/caption_datasets'
    if not os.path.exists(os.path.abspath('.') + annotation_folder):
        os.system('wget -p /coco_dataset/ http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip')
        os.system('unzip caption_datasets.zip')
        os.remove('caption_datasets.zip')
        print("captions have been downloaded")
    else:
        print("captions have already exist")
        
# Yan Liu's Master Project - Interactive image description generation system

This repository is created for my master project in the university of manchester. It contains the source code and some important experimental data that shown in my dissertation.

If you have any question, please email me: [YanLiu](mailto:yan.liu-25@student.manchester.ac.uk")

The demo website for my project is: http://78.141.226.122:5000/.

## 1. Project Structure

```txt
.master_project_image_caption   // the root directory
├── coco_dataset
│   ├── caption_datasets        // (nedd download) annotation (train,val,test set) of the coco 2014 dataset
│   ├── train2014               // (nedd download) coco 2014 train set (images)
│   ├── val2014                 // (nedd download) coco 2014 validation set (images)
├── prepare
│   ├── data_prprocess.py       // pre-process annotation data
│   ├── word2vec.py             // build vocabulary and convert sentence to vector
│   ├── image2hdf5.py           // store all the images into hdf5 file (very large)
├── models
│   ├── encoder.py              // Encoder which contains Resnet152 model
│   ├── decoder_Attention.py    // Decoder which contains LSTM and Attention model
├── train
│   ├── dataset.py              // input batch data into the model
│   ├── train.py                // train the model. Including loss, accuracy, validation, etc.
├── evaluate
│   ├── evaluate.py             // evaluate the performance of the trained model based on COCO test set
│   ├── pycocoevalcap           // Official evaluation tool (calculate bleu, metor, spice, cider, etc)
├── test
│   ├── discrible.py            // predict the input image's description
├── saved_data
│   ├── trained_models          // folder to save the trained model
│   ├── word2index              // folder to save vocabulary
│   ├── cap2vec                 // folder to save annotations' vectors format
│   ├── evaluation              // save prediction results and true labels for test set
│   ├── hdf5_images             // images hdf5 format:test_images.hdf5, train_images.hdf5, val_images.hdf5
├── static
│   ├── css                     // css file for web page
│   ├── js                      // javascript file for web page
│   ├── demo_img                // 20 examples random selected form dataset or web
│   ├── result_img              // store predicted result of uploaded (test) image
│   ├── uploaded_img            // store uploaded(test) images
├── templates
│   ├── index.html              // HTML file for home page
│   ├── ...
│   ├── upload.html             // HTML file for uploading images and testing them
.
```

## 2. Train the model

### 2.1 Download data

You should download below COCO 2014 dataset and its annotation.

| Data name | Download Link |
| ------------ | ------------- |
| COCO image dataset | [train set](http://images.cocodataset.org/zips/train2014.zip), [val set](http://images.cocodataset.org/zips/val2014.zip)|
| COCO annotation set | [Dowload](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) |

Then put downloaded COCO `train2014` and `val2014` folder in coco_dataset, just like above structure.

Then put downloaded `caption_datasets` in coco_dataset folder, just like above structure.

### 2.2 Necessary development environment

So far, you have prepared the data, then you nedd to install necessary enviroment:

| python library | version |
| ------------ | ------------- |
| python | 3.7 or 3.68 |
| pytorch | 1.5 or 1.40 |
| torhvision| 0.6 |
| CUDA| 10.1 |
| cudNN| 7 |
| java| 1.8 |
| scikit-imag| 0.17.2 |
| Flask| 1.1.2 |
| progress| 0.17.2 |
| tqdm| 4.46.1 |
| nltk|3.5|
|h5py|2.10.0|
|imageio|2.80|
|numpy|1.18.5|
|matplotlib|3.2.2|
|etc.|install in python3.7|

Ok, now, you can run below comand step by step to train the model ! Also, you can open the train.py file and adjust some parameters or setting to train. It took more than an hour to train an epoch on an RTX 2080Ti GPU.

```txt
python3 main.py -c data_prepare

python3 main.py -c train
```

## 3. Evaluate the trained model

### 3.1 use the model trained by yourself or provided by me

If you dont like to train a new model and still want to do the this work, you can download pre-trained models below provided by me and pick one to do the next job. Besides, you also need to download the vocabulary.The optimal_model is recommended.

| Model | Downlaod Link |
| ------------ | ------------- |
| optimal_model | [Dowload](https://livemanchesterac-my.sharepoint.com/:u:/g/personal/yan_liu-25_student_manchester_ac_uk/EVJB1vVWpUJDgl84HLkLw9gB8-8pfkFvmFBH5ARX1ViwPA?e=nYohr9)|
| fine_tuned model with resnet101|[Dowload](https://livemanchesterac-my.sharepoint.com/:u:/g/personal/yan_liu-25_student_manchester_ac_uk/EZq4RxbjqApJuVeIvrnyRTYBHl73q8fLu4ZLTpJe40vpGA?e=Fui6tA)|
| fine_tuned model with resnet152 | [Dowload](https://livemanchesterac-my.sharepoint.com/:u:/g/personal/yan_liu-25_student_manchester_ac_uk/EVJB1vVWpUJDgl84HLkLw9gB8-8pfkFvmFBH5ARX1ViwPA?e=nYohr9)|
| resnet101 as encoder without finetuning (result not good)| [Dowload](https://livemanchesterac-my.sharepoint.com/:u:/g/personal/yan_liu-25_student_manchester_ac_uk/ETm2iyyPGpNPm8sdQBaw7-0BtaM5VXYoivgfv0PE90eqmw?e=sdYKXg)|
| resnet152 as encoder without finetuning (result not good)| [Dowload](https://livemanchesterac-my.sharepoint.com/:u:/g/personal/yan_liu-25_student_manchester_ac_uk/EVGz8wPDS6dCljjZgOnpWHgBLSNbWht6xXJ4EzClgPhVuQ?e=GOcUPe)|
| vocabulary| [Must Dowload](https://livemanchesterac-my.sharepoint.com/:u:/g/personal/yan_liu-25_student_manchester_ac_uk/EWjLVuz-lXdOuOm902UiRucBZzyxTuoaCvDZUi1ij72KBg?e=XcXl3E)|

**Make sure you put the dowloaded model in the saved_data/trained_models, put the vocabulary in the saved_data/word2index and change the model's path and vocabulary's path in `evaluate.py`**.
Then you can run following command to evaluate model on coco test set, we will get the scores of bleu1-4, metor, cider, rouge, and spice.

```txt
python3 main.py --evaluate 3   # 3 means  beam search size is 3
```

This also taks a lot of time (almost 30 minutes)

### 3.1 use generated results instead of the model

Of course you can download the pre-saved prediction results and ture labels to save a lot of time !
For example, we donwload the results (from below link) for the model with resnet152 as encoder without finetuning and beam search size is 3. (make sure you have vocabulary)

| Result | Downlaod Link |
| ------------ | ------------- |
| evaluation result | [Dowload](https://livemanchesterac-my.sharepoint.com/:f:/g/personal/yan_liu-25_student_manchester_ac_uk/EtR4UNjEI-NAvU-DMfkk9YcBJb0CffkQuLvs5FBlOh8a1g?e=6udXhL)|

Download the `resnet_152_bs3_test_hyp.json` and `resnet_152_bs3_test_ref.json`. Put them in the saved_data/evaluation/ foloder. Run the command below.

```txt
python3 main.py --evaluate pre_save
```

### 4. Generate description for an image (Test the model)

There are two ways to get the predicted description for your input image! One is using the terminal command, another is using the website I developed in this project.

#### Way-1: on terminal

Assume you already have a trained model (train yourself or download mine) and a vocabulay.

Just run below command:

```txt
python3 main.py --test 'xxx/xxx/xx'   # xxx/xxx/xx means the path of tested image
```

#### Way-2: use website (recommended)

Make suire the model's path is right in `run_app.py file`. Start the server and you can to open http://0.0.0.0:5000/ to use this interactive web site.

```txt
Python3 run_app.py
```

## 5. Acknowledgement

I would like to acknowledge my supervisor Dr. Aphrodite Galata for providing guidance, moral support, and feedback throughout this project. She made very meaningful suggestions for my project and guided me on how to write this dissertation.
I would aslo like to thank [Kelvin Xu](https://arxiv.org/abs/1502.03044) for his paper, Sagar for the Pytorch Tutorial and Codepen io for its html and css font.

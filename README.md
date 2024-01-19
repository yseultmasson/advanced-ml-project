# ENSAE Paris | IP Paris

## Advanced Machine Learning

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/LOGO-ENSAE.png/900px-LOGO-ENSAE.png" width="200">

## Topic : Neural Style Transfer as a Data Augmentation Method🎨

This GitHub repository constitutes the coding part of our assignment for the course "Advanced Machine Learning". It consists of the reproduction and study of the work "STaDA: Style Transfer as Data Augmentation", written by Xu Zheng et al. in January 2019. The article can be read here : https://arxiv.org/abs/1909.01056.
Our report can be found here : (lien vers le rapport, ou dire que le rapport est aussi présent dans le repository)

### Realised by : 

* Matthieu BRICAIRE
* Yseult MASSON
* Rayan TALATE

### Teacher : 

* Austin STROMME

#### Academic year: 2023-2024

October 2023 - January 2024.


## Table of Contents

1. [Overview](#overview)
2. [File Description](#description)
    1. [File Tree](#tree)
    2. [Dependencies](#dependencies)
3. [How to generate images?](#generation)
    1. [For the descriptive approach](#descriptive_generation)
    2. [For the generative approach](#generative_generation)
4. How was this repository created? (#creation)
    1. [For the descriptive approach](#descriptive_creation)
    2. [For the generative approach](#generative_generation)
5. [Acknowledgements](#ack)

## Overview <a name="overview"></a>

Neural Style Transfer (NST) is about manipulating digital images in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation (more specifically, adding the style of an image to the content of another image).

In this repository, we explored and coded the two main ways described by Xu Zheng et al. to conduct Neural Style Transfer, namely the descriptive approach and the generative approach. We then used these models to augment the Caltech101 dataset and compared the effect of different augmentations on a classification task.  These analyses are described in-depth in our report.


## File Description <a name="description"></a>
### File Tree <a name="tree"></a>

    advanced-ml-_project
        │   README.md
        │   requirements.txt
        │
        ├───classification
        │   │   augment_data.py
        │   │   classification.py
        │   │   test_results.ipynb
        │   │   train_test_split.py
        │   ├───checkpoints
        |   ├───data
        │   ├───histories
        │   └───results
        |
        └───style_transfer
            │   image_transformer_net.py
            │   descriptive_generation.py  <-- the main python file for the descriptive approach.
            │   style_transfer.py <-- file to apply a trained generative model to some images.
            │   train.py <-- file to train a generative model from scratch.
            │   utils.py
            │   vgg16.py
            ├───images  <-- Where all images are stored.
            │   ├───base_images
            │   ├───output_images
            │   └───style
            └───models <-- Where the 3 models for the generative approach with our 3 base styles are stored.


## Running the code<a name="run_code"></a>

First, clone the entire repository:
```bash
git clone https://github.com/yseultmasson/advanced-ml-project/
```

If you use conda, create an empty environment :
```bash
conda create -n env_name python==3.9.13
```

Then, install the requirements:
```bash
pip install -r requirements.txt
```
Note that in the version of pytorch installed is optimized for CPU. If you want to install another version that works for GPU, follow the instuctions from [the PyTorch website](https://pytorch.org/get-started/locally/).

The style transfer and classification parts are independant. You can run only the style transfer scripts or only the classification scripts.

### How to generate images?<a name="generation"></a>

Head over to the "style_transfer" directory:
```bash
cd style_transfer
```

#### For the descriptive approach : <a name="descriptive_generation"></a>

Once this is done, you may run descriptive_generation.py. The generated image will be saved in the `descriptive_generation` folder inside `images\output-images`.

The repository already provides 3 style images and 10 base images. If you want to use other images, upload them in the folders `images\base_images` or `images\style` according to the type of image, and change the following lines in descriptive_generation.py :

```
CONTENT_IMAGE = r"bus.jpg" <-- put the name of the image you want as a base image (line 261)
STYLE_IMAGE = r"starry_night.jpg" <-- put the name of the image you want as a style image (line 262)
```


#### For the generative approach : <a name="generative_generation"></a>
If you want to evaluate one of the provided models on the available test images, you can run :
```bash
python style_transfer.py --model_path models/mosaic_2_epochs_82783_samples_2_1.0_cttwght.model --source images/base_images --output images/output_images/generative_st
```
You may adjust the model you want to you use as you see fit. If you want to evaluate a model on other images than the ones provided here, you just need to create a new folder inside the "images" folder, store your images inside, and adjust te command to execute.

If you want to train a new model from scratch, you first need to collect the training data and store it in a specific folder, inside the "images" folder. To train our models, we used the COCO 2014 training dataset, which you can download here : https://cocodataset.org/#download (we only used the file named '2014 Train images [83K/13GB]', which corresponds to the 'train2014.zip' file). Once you downloaded this file, you can create a 'train_data' folder inside the 'images' one, and extract the content of the 'train2014.zip' file inside it.

Once this is done, your "images" folder should look like this :

    images
       ├───base_images
       ├───output_images
       ├───style
       └───train_data
                └───train2014
                        ├───image1.jpg
                        └───image2.jpg

You may use another dataset of your choice, as long as the resulting folder has the same architecture as the above one.

Once you have your data ready to use, you can run the following command :
```bash
python train.py --style-image images/style/mosaic.jpg --dataset images/train_data
```

You may adjust the style image you want to use, as well as the script's available arguments as you see fit.

### Data augmentation and classification

In order to run the classification experiments, you first need to download the Caltech101 dataset (https://data.caltech.edu/records/mzrjq-6wc02), move the folder "101_ObjectCategories" into `classification/data`, and remove the subfolder `BACKGROUNG_Google` (which contains background clutter and is not interesting for our analysis) from `classification/data/101_ObjectCategories`. Then, create the train, validation and test sets by running:

```bash
python train_test_split.py
```

To augment the train set with the augmentation strategy `augmentation` (which must be 'flip', 'starry_night', 'mosaic', 'tapestry', or any combination of those separated by a '-', for example 'flip-mosaic-starry_night'), do:

```bash
python augment_data.py --original_train_dir data/train_set -a augmentation
```

Then, to fine-tune a VGG16 network on an augmented train set, run:

```bash
python classification.py --train_dir data/train_set_augmentation --val_dir data/val_set -a augmentation
```
and replace `train_set_augmentation` and `augmentation` by the desired augmentation strategy.

To train the model on the original train set, run:
```bash
python classification.py --train_dir data/train_set --val_dir data/val_set -a no_aug
```

The evalutation of the models on the test set are done in `test_results.ipynb`.


## How was this repository created?<a name="creation">

### Descriptive approach <a name="descriptive_creation">
The descriptive approach consists of only one script, "descriptive_generation.py". It is a commented and lightly modified version of the script "NST.py", available here: https://github.com/nazianafis/Neural-Style-Transfer/tree/main.
A lot of comments have been written to make sure the code can be somewhat understandable to somebody who just read the original paper from Gatys et. al. As for the modifications, they were mainly caused by the fact that we wanted to use our own implementation of the VGG16 network instead of the VGG19 network initially used. We also made some changes to the file tree of the repository, because we needed to harmonize this part with the rest of our repository. We did not feel the need to heavily change this code, because it does not represent the core of our work. This part was just needed to make a speed comparison with the code of the generative approach, which would be used much more extensively.

### Generative approach <a name="generative_creation">

The generative approach relies on several scripts, namely : "image_transformer_net.py", "style_transfer.py", "train.py", "utils.py" and "vgg16.py".
To implement it, we started from an existing implementation of the method, available here : https://github.com/dxyang/StyleTransfer/tree/master.
Our "image_transformer_net.py" script is a commented version of the "network.py" script from the existing implementation. We also reused the main functions available in "utils.py".
Our "style_transfer.py" and "train.py" scripts are inspired from the "style.py" script, but have been largely modified and customized to our needs.
Our "vgg16.py" script was entirely built by ourselves, to meet our needs (in particular, to work for both studied style transfer methods).

### Classification

In order to fine-tune the VGG16 model on our dataset, we followed the [Tensorflow tutorial on transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning), and adapted it to our needs.

## Acknowledgements <a name="ack"></a>

These are some of the resources we used or consulted while working on this project:


* The original paper on neural style transfer by [Gatys et al](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
* The repository used to code the descriptive approach of style transfer, by [nazianafis](https://github.com/nazianafis/Neural-Style-Transfer/tree/main)
* The repository used to implement the generative approach of style transfer, by [dxyang](https://github.com/dxyang/StyleTransfer/tree/master)



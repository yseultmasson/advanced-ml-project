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
and install the requirements:
```bash
pip install requirements.txt
```

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
Say we need to download data first, where?
Run the following command in a terminal:

```bash
python style_transfer.py --model_path models/starry_night_2_epochs_82783_samples_2_1.0_cttwght.model --source data/test_images --output data/test_results
```

fill how to use the code with other base and style images.

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
The descriptive approach consists of only one script, "descriptive_generation.py". It is a commented and lightly modified version of the script "NST.py", available here: https://github.com/nazianafis/Neural-Style-Transfer/tree/main
A lot of comments have been written to make sure the code can be somewhat understandable to somebody who just read the original paper from Gatys et. al. As for the modifications, they were mainly caused by the fact that we wanted to use our own implementation of the VGG16 network instead of the VGG19 network initially used. We also made some changes to the file tree of the repository, because we needed to harmonize this part with the rest of our repository. We did not feel the need to heavily change this code, because it does not represent the core of our work. This part was just needed to make a speed comparison with the code of the generative approach, which would be used much more extensively.

### Generative approach <a name="generative_creation">

to fill

### Classification

In order to fine-tune the VGG16 model on our dataset, we followed the [Tensorflow tutorial on transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning), and adapted it to our needs.



## Acknowledgements <a name="ack"></a>

These are some of the resources we used or consulted while working on this project:


* The original paper on neural style transfer by [Gatys et al](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
* The repository used to code the descriptive approach of style transfer, by [nazianafis](https://github.com/nazianafis/Neural-Style-Transfer/tree/main) 



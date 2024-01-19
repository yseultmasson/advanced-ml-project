# ENSAE Paris | Institut Polytechnique de Paris

## Advanced Machine Learning

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/LOGO-ENSAE.png/900px-LOGO-ENSAE.png" width="300">

## Topic : Neural Style Transfer as a Data Augmentation Method🎨

This GitHub repository constitutes the coding part of our assignment for the course "Advanced Machine Learning". It consists of the reproduction and study of the work "STaDA: Style Transfer as Data Augmentation", written by Xu Zheng et al. in January 2019. The article can be read here : https://arxiv.org/abs/1909.01056
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

Neural Style Transfer (NST) is about manipulating digital images in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation.

In this repository, we explored and coded the two main ways described by Xu Zheng et al. to conduct Neural Style Transfer, namely the descriptive approach and the generative approach. Read our article for an in-depth description of those two approaches. 


## File Description <a name="description"></a>
### File Tree <a name="tree"></a>
    advanced-ml-_project
        ├── classification
            ├── histories
            ├── results
            ├── augment_data.py <--      
            ├── classification.py <--
            ├── test_results.ipynb
            └── train_test_split.py
        
        ├── style_transfer  <-- Where the 3 models for the generative approach with our 3 base styles are stored.
            ├── images <-- Where all images are stored.
            |   ├── base-images
            |   ├── style
            |   ├── output-images 
            ├── descriptive_generation.py  <-- the main python file for the descriptive approach.
            ├── image_transformer_net.py <--      
            ├── style_transfer.py
            ├── train.py
            ├── utils.py
            └── vgg16.py  
        ├── README.md
        └── requirements.txt
        

### Dependencies <a name="dependencies"></a>
*    Python 3.9+
*    Libraries: torch, os, numpy, cv2, matplotlib, torchvision, argparse


## How to generate images?<a name="generation"></a>

First, you need to install the requirements for the codes to work. Clone the entire repository:
```bash
git clone https://github.com/yseultmasson/advanced-ml-project/
```

 then run the following command in a terminal:
```bash
python pip install requirements.txt
```

Once this is done, head over to the "style_transfer" directory:
```bash
python cd=style_transfer
```

### For the descriptive approach : <a name="descriptive_generation"></a>

Once this is done, you may run descriptive_generation.py. Find your generated image in the `descriptive_generation` folder inside `images\output-images`.

The repository already provides 3 style images and 10 base images. If you want to use other images :
Move the images you want in the according folder between images\base_images and images\style. Then, change the following lines in descriptive_generation.py :

```
CONTENT_IMAGE = r"bus.jpg" <-- put the name of the image you want as a base image (line 261)
STYLE_IMAGE = r"starry_night.jpg" <-- put the name of the image you want as a style image (line 262)
```


### For the generative approach : <a name="generative_generation"></a>
You may run the following command in a terminal:

```bash
python style_transfer.py --model_path models/starry_night_2_epochs_82783_samples_2_1.0_cttwght.model --source data/test_images --output data/test_results
```

fill how to use the code with other base and style images.

## How was this repository created?<a name="creation">

### For the descriptive approach <a name="descriptive_creation">
The descriptive approach consists of only one script, "descriptive_generation.py". It is a commented and lightly modified version of the script "NST.py", available here: https://github.com/nazianafis/Neural-Style-Transfer/tree/main
A lot of comments have been written to make sure the code can be somewhat understandable to somebody who just read the original paper from Gatys et. al. As for the modifications, they were mainly caused by the fact that we wanted to use our own implementation of the VGG16 network instead of the VGG19 network initially used. We also made some changes to the file tree of the repository, because we needed to harmonize this part with the rest of our repository. We did not feel the need to heavily change this code, because it does not represent the core of our work. This part was just needed to make a speed comparison with the code of the generative approach, which would be used much more extensively.

### For the generative approach <a name="generative_creation">

to fill

## Acknowledgements <a name="ack"></a>

These are some of the resources we used or consulted while working on this project:


* The original paper on neural style transfer by [Gatys et al](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
* The repository used to code the descriptive approach of style transfer, by [nazianafis](https://github.com/nazianafis/Neural-Style-Transfer/tree/main) 



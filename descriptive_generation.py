# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:57:18 2024

This code aims to reproduce the article "Image Style Transfer Using Convolutational Neural Networks", written by Gatys et. AL in 2016. Refer to it for more detail regarding some results.
Link to the article: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
"""


import os
import numpy as np
from tqdm import tqdm

import tensorflow as tf #if using GPU on windows: pip install tensorflow==2.10
from tensorflow import keras
from tensorflow.keras.applications import vgg16


from tensorflow.python.client import device_lib
device_lib.list_local_devices()

#%%
# Generated image size
RESIZE_HEIGHT = 256

NUM_ITER = 50

# Weights of the different loss components
CONTENT_WEIGHT = 1 # 8e-4
STYLE_WEIGHT = 1e5 # 8e-4
#content to style ration is of 1e-5

# The layer to use for the content loss. The latter layers do a better job at extracting the semantic content of an image without being subservient to detailed pixel information.
#This is equivalent to the "conv5" layer at the upper right of the Fig.2 of the article.
CONTENT_LAYER_NAME = "block5_conv2" 

# List of layers to use for the style loss.
# Those are equivalent to the 5 convolutional layers at the left of the Fig.2 of the article
STYLE_LAYER_NAMES = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

def get_result_image_size(image_path:str, result_height:int) -> (int,int) :
    """
    This function will be used to fix the size of the result image beforehand. 
    It needs the path of the image to extract its width and height, and the height the user wants the result image to be.
    The result width is calculated using the base size and the result height to preserve the size ratio of the image.
    
    Parameters
    ----------
    image_path : string
        The path to the image.
        
    result_height : int
        The height of the result image. Chosen by the user

    Returns
    -------
    result_height : int
        The same height that was passed as argument.
    result_width : int
        The width of the result image. Computed in a way that makes sure the input and output images have the same width/height ratio.

    """
    image_width, image_height = keras.preprocessing.image.load_img(image_path).size
    result_width = int(image_width * result_height / image_height)
    return result_height, result_width

def preprocess_image(image_path:str, target_height:int, target_width:int) -> tf.Tensor :
    """
    Turns an input image into a tensor that can be fed to our model, which uses the VGG16 network.
    This network has its own preprocessing function, which simplifies the process.

    Parameters
    ----------
    image_path : str
        the path to the image.
    target_height : int
        the height of the result image.
    target_width : int
        the width of the result image.

    Returns
    -------
    a tf.Tensor object.

    """
    img = keras.preprocessing.image.load_img(image_path, target_size = (target_height, target_width))
    arr = keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis = 0)
    arr = vgg16.preprocess_input(arr)
    return tf.convert_to_tensor(arr)

def get_model() -> keras.Model:
    """
    Builds the VGG16 model, loaded with pre-trained ImageNet weights.
    We want this model to keep the record of the outputs of some 'key' layers.

    Returns
    -------
    a keras.Model object, ready to be used.

    """
    model = vgg16.VGG16(weights = 'imagenet', include_top = False)

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    return keras.Model(inputs = model.inputs, outputs = outputs_dict)

def get_optimizer() -> keras.src.optimizers.adam.Adam :
    """
    Loads the Adam Optimizer. Gatys et.Al preferred the L-BFGS algorithm, but due to the fact that L-BFGS uses the second order derivative, its computation cost is higher.
    Since the optimizer is used at every iteration, we deem this computation cost difference significant enough to justify a change in the optimizer.

    Returns
    -------
    keras.src.optimizers.adam.Adam
        The Adam Optimizer with specific hyperparameters.
    """
    return keras.optimizers.Adam(keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 8.0, decay_steps = 445, decay_rate = 0.98))

# Computing the losses 

def gram_matrix(x) -> tf.Tensor:
    """
    Computes the gram matrix of the features. This gram matrix is used in the computation of the style loss.

    Parameters
    ----------
    x : tf.Tensor

    Returns
    -------
    gram : tf.Tensor
        X^T*X

    """
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style_features:tf.Tensor, combination_features:tf.Tensor, combination_size:int) -> int :
    """
    Uses the gram_matrix function to calculate the style loss. Note that this loss will be computed for every layer in the list "STYLE_LAYER_NAMES", defined at the beginning of the script.

    Parameters
    ----------
    style_features : tf.Tensor
        The style representation of the generated image at this point of the algorithm, as a tensor.
    combination_features : tf.Tensor
        The style representation of the input image, as a tensor.
    combination_size : int
        The number of layers.

    Returns
    -------
    int
        The style loss, see equation (4) of the article for more details regarding the computation.

    """
    S = gram_matrix(style_features)
    C = gram_matrix(combination_features)
    channels = style_features.shape[2]
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (combination_size ** 2))

def total_style_loss(style_features, combination_features, combination_size) -> int:
    """
    Averages the style losses for every relevant layer to calculate the total style loss.
    The parameters are the same as in the 'style_loss' function

    Returns
    -------
    loss_style : int
        The average of the style losses.

    """
    loss_style = 0
    for layer_name in STYLE_LAYER_NAMES:
        style_feature = style_features[layer_name][0]
        combination_feature = combination_features[layer_name][0]
        loss_style += style_loss(style_feature, combination_feature, combination_size) / len(STYLE_LAYER_NAMES)

    return loss_style



def content_loss(content_features:tf.Tensor, combination_features:tf.Tensor) -> int :
    """
    Computes the content_loss between the feature representation of the original image and the generated image at a specified layer of the network, specified at the beginning of the script by "CONTENT_LAYER_NAME".
    
    
    Parameters
    ----------
    content_features : tf.Tensor
        The feature representation of the original image at this point of the algorithm, as a tensor.
    combination_features : tf.Tensor
        DESCRIPTION.

    Returns
    -------
    int
        The content loss, see equation (1) for more details regarding the computation.

    """
    original_image = content_features[CONTENT_LAYER_NAME]
    generated_image = combination_features[CONTENT_LAYER_NAME]

    return tf.reduce_sum(tf.square(generated_image - original_image)) / 2


def compute_loss(feature_extractor:keras.Model, combination_image:tf.Variable, content_features:tf.Tensor, combination_features:tf.Tensor) -> int:
    """
    computes the total loss of the generated image and the original input. 
    This function requires two hyperparameters, "STYLE_WEIGHT" and "CONTENT_WEIGHT".
    See the functions 'content_loss', 'style_loss' and 'total_style_loss' for more details.
    
    Parameters
    ----------
    feature_extractor : keras.Model
        The model that will receive the image. In our case, it will be a VGG16 architecture, trained on ImageNet weights.
    combination_image : tf.Variable
        The tensor of the generated image
    content_features : TYPE
        The feature representation of the original image at this point of the algorithm, as a tensor.
    style_features : TYPE
        The style representation of the original image at this point of the algorithm, as a tensor.

    Returns
    -------
    int
        The total loss, as computed in (7)

    """
    combination_features = feature_extractor(combination_image)
    loss_content = content_loss(content_features, combination_features)
    loss_style = total_style_loss(style_features, combination_features, combination_image.shape[1] * combination_image.shape[2])

    return CONTENT_WEIGHT * loss_content + STYLE_WEIGHT * loss_style



def deprocess_image(tensor:tf.Tensor, result_height:int, result_width:int) -> np.ndarray:
    """
    Turns a tf.Tensor into an RGB matrix, which can then be saved as an image. This funtion is called in save_result below.

    Parameters
    ----------
    tensor : tf.Tensor
        a Tensor that will be turned into an image.
    result_height : int
        self-explanatory.
    result_width : int
        self-explanatory.
    name : str

    Returns
    -------
    np.ndarray
        3-dimensional array corresponding to the RGB values of the image.

    """
    tensor = tensor.numpy()
    tensor = tensor.reshape((result_height, result_width, 3))

    # Remove zero-center by mean pixel
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.680

    # 'BGR'->'RGB'
    tensor = tensor[:, :, ::-1]
    return np.clip(tensor, 0, 255).astype("uint8")

def save_result(generated_image:tf.Tensor, result_height:int, result_width:int, name:str) -> None :
    """
    Saves a tf.Tensor as an image in the working directory, or in another directory to be specified.

    Parameters
    ----------
    generated_image : tf.Tensor
        a Tensor that will be turned into an image.
    result_height : int
        self-explanatory.
    result_width : int
        self-explanatory.
    name : str
        The name of the image, as well as the full path leading to it if needed.


    """
    img = deprocess_image(generated_image, result_height, result_width)
    keras.preprocessing.image.save_img(name, img)

if __name__ == "__main__":
    # Get the path of the working directory. The base image (whose content will be reproduced in a different style) and the style image have to be put in specific subfolders.
    path = os.path.abspath(os.getcwd())
    
    #retrieve the base image and the style image. 
    content_image_path = keras.utils.get_file(path + '\dataset\paris.jpg', 'https://i.imgur.com/F28w3Ac.jpg')
    style_image_path = keras.utils.get_file(path + '\dataset\starry_night.jpg', 'https://i.imgur.com/9ooB60I.jpg')
    
    result_height, result_width = get_result_image_size(content_image_path, RESIZE_HEIGHT)
    
    file=open('progress.txt','w') #This file will be used to monitor the progress of our generation through tqdm.
    # print("result resolution: (%d, %d)" % (result_height, result_width))

    # Preprocessing
    content_tensor = preprocess_image(content_image_path, result_height, result_width)
    style_tensor = preprocess_image(style_image_path, result_height, result_width)
    generated_image = tf.Variable(tf.random.uniform(style_tensor.shape, dtype=tf.dtypes.float32))

    # Building the model
    model = get_model()
    optimizer = get_optimizer()
    # print(model.summary())

    content_features = model(content_tensor)
    style_features = model(style_tensor)

    # Optimize result image through gradient descent at each step
    
    for iter in tqdm(range(NUM_ITER),file=file):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, generated_image, content_features, style_features)

        grads = tape.gradient(loss, generated_image)

        # print("iter: %4d, loss: %8.f" % (iter, loss))
        optimizer.apply_gradients([(grads, generated_image)])

        if (iter + 1) % 600 == 0:
            name = "result/generated_at_iteration_%d.png" % (iter + 1)
            save_result(generated_image, result_height, result_width, name)

    name = "result/result_%d_%f_%f.png" % (NUM_ITER, CONTENT_WEIGHT, STYLE_WEIGHT)
    save_result(generated_image, result_height, result_width, name)
    file.close()

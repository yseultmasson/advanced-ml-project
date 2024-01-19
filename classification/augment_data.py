# This script is used to augment the images of a training set.

# Example of use, with "mosaic" augmentation:
# python augment_data.py --original_train_dir data/train_set -a mosaic
# With "mosaic" and "flip" augmentation:
# python augment_data.py --original_train_dir data/train_set -a mosaic-flip

from PIL import Image
import numpy as np
import os
import shutil
from argparse import ArgumentParser
from tqdm import tqdm
from torchvision import transforms
import torch
import os
from typing import Optional
import sys

# Add path 
sys.path.append('../style_transfer/')

from image_transformer_net import ImageTransformNet
from utils import *



def style_transfer(image: Image.Image, device: torch.device, model: torch.nn.Module, img_size: int = 180) -> Image.Image:
    """
    Apply style transfer to the input image using the given model.

    Parameters:
    - image (PIL.Image.Image): The original image.
    - device (torch.device): The device on which to perform the style transfer (e.g., 'cuda' for GPU or 'cpu').
    - model (torch.nn.Module): The pre-trained style transfer model.
    - img_size (int): The size of the output image. Default is 180.

    Returns:
    - PIL.Image.Image: The stylized image.
    """

    # Preprocess image
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),          # Scale shortest side to image_size
        transforms.CenterCrop(img_size),                  # Crop center image_size out
        transforms.ToTensor(),                            # Turn image from [0-255] to [0-1]
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)            # Normalize with ImageNet values
    ])

    image = image_transform(image).unsqueeze(0).to(device)

    # Perform style transfer
    image = model(image).cpu().data[0]

    # Denormalize the output image
    mean = np.array(IMAGENET_MEAN).reshape((3, 1, 1))
    std = np.array(IMAGENET_STD).reshape((3, 1, 1))
    image = image.clone().numpy()
    image = ((image * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")

    # Convert the numpy array to a PIL Image
    image = Image.fromarray(image)

    return image



def augment_image(image: Image.Image, aug: str, device: torch.device, model: Optional[torch.nn.Module] = None, img_size: int = 180) -> Image.Image:
    """
    Apply specified image augmentation to the input image

    Parameters:
    - image (PIL.Image.Image): The original image.
    - aug (str): The type of augmentation to be applied. Options: 'flip', 'starry_night', 'tapestry', 'mosaic'.
    - device (torch.device): The device on which to perform the style transfer (e.g., 'cuda' for GPU or 'cpu').
    - model (torch.nn.Module): The pre-trained style transfer model if style transfer is used. Default is None.
    - img_size (int): The size of the output image. Default is 180.

    Returns:
    - PIL.Image.Image: The augmented image.
    """

    # Augment image according to the type of augmentation needed

    if aug == 'flip':
        # Flip the image horizontally
        augmented_img = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    elif aug in ['starry_night', 'tapestry', 'mosaic']:
        # Apply style transfer augmentation
        augmented_img = style_transfer(image, device, model, img_size)

    else:
        raise ValueError("Invalid aug value. It must be either 'flip', 'starry_night', 'tapestry', or 'mosaic'.")

    return augmented_img



def main(args):
    """
    Perform data augmentation on the training set using specified augmentation types.
    Save the images (original and augmented) in a new folder.
    """

    # Set the arguments
    original_train_path = args.original_train_dir
    augmentation = args.augmentation
    img_size = args.img_size

    # Split the augmentation string into a list of individual augmentations
    augs = augmentation.split('-')

    #Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained models for style transfer augmentations
    models = dict()
    for aug in augs:
        if aug in ['starry_night', 'mosaic', 'tapestry']:
            models[aug] = ImageTransformNet().to(device)
            models[aug].load_state_dict(torch.load(f'../style_transfer/models/{aug}_2_epochs_82783_samples_2_1.0_cttwght.model', map_location=device))
        else:
            models[aug] = None


    # Set the path to the train directory
    train_path = f'{original_train_path}_{augmentation}'
    os.mkdir(train_path)

    # Iterate through each category folder in the original training set
    for category_folder in tqdm(os.listdir(original_train_path), desc='Augmentation progress'):

        category_path = os.path.join(original_train_path, category_folder)

        # Make a path for this category in the augmented train set
        os.mkdir(os.path.join(train_path, category_folder))

        # List all images in the category folder
        images = os.listdir(category_path)

        for image in images:
            # Path of the original image
            source_path = os.path.join(category_path, image)

            # Path for the original and augmented images in the augmented train set
            destination_path = os.path.join(train_path, category_folder, image.split(".")[0])

            # Copy the original image
            shutil.copyfile(source_path, f"{destination_path}.jpg")

            try:
                #Open the original image
                img_to_augment = Image.open(source_path)

                for aug in augs:
                    # Augment the image
                    augmented_img = augment_image(img_to_augment, aug, device, models[aug], img_size)
                    
                    with open(f'{destination_path}_{aug}.jpg', 'wb') as file:
                        # Save the augmented image
                        augmented_img.save(file)

            except Exception as e:
                print(e)


            
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--original_train_dir', help='Not augmented train set path')
    parser.add_argument('-a', '--augmentation', help='Augmentation types, separated by - (ex : starry_night-mosaic-flip)', type=str)
    parser.add_argument('--img_size', help='Image size', default=180)
    
    args = parser.parse_args()

    main(args)

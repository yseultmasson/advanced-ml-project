from PIL import Image
import os
import shutil
from argparse import ArgumentParser
from tqdm import tqdm
from torch.autograd import Variable
from torchvision import transforms
import torch
import os
import re

from image_transformer_net import ImageTransformNet
from utils import *


def style_transfer(image, device, model, img_size=180):

    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),          # scale shortest side to image_size
        transforms.CenterCrop(img_size),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)      # normalize with ImageNet values
    ])

    image = image_transform(image).unsqueeze(0).to(device)
    image = model(image).cpu().data[0]
    mean = np.array(IMAGENET_MEAN).reshape((3, 1, 1))
    std = np.array(IMAGENET_STD).reshape((3, 1, 1))
    image = image.clone().numpy()
    image = ((image * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    image = Image.fromarray(image)

    return image


def augment_image(image, aug, device, model, img_size=180):

    if aug == 'flip':
        augmented_img = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    elif aug in ['starry_night', 'tapestry', 'mosaic']:
        augmented_img = style_transfer(image, device, model, img_size)

    else:
        raise ValueError("Invalid aug value. It must be either 'flip', 'starry_night', 'tapestry' or 'mosaic'.")

    return augmented_img


def main(args):

    #Set the arguments
    original_train_path = args.original_train_dir
    augmentation = args.augmentation
    img_size = args.img_size

    augs = augmentation.split('-')

    #Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = dict()

    for aug in augs:
        if aug in ['starry_night', 'mosaic', 'tapestry']:
            models[aug] = ImageTransformNet().to(device)
            models[aug].load_state_dict(torch.load(f'models/{aug}_2_epochs_82783_samples_2_1.0_cttwght.model'))
        else:
            models[aug] = None


    # Set the path to the train directory
    train_path = f'{original_train_path}_{augmentation}'
    os.mkdir(train_path)

    # Iterate through each category folder
    for category_folder in tqdm(os.listdir(original_train_path), desc='Augmentation progress'):

        category_path = os.path.join(original_train_path, category_folder)

        #Make a path for this category for the augmented train set
        os.mkdir(os.path.join(train_path, category_folder))

        # List all images in the category folder
        images = os.listdir(category_path)

        for image in images:
            #Path of the image
            source_path = os.path.join(category_path, image)
            #Where to save the original and augmented images for the augmented train set
            destination_path = os.path.join(train_path, category_folder, image.split(".")[0])

            #Copy the original image
            shutil.copyfile(source_path, f"{destination_path}.jpg")

            try:
                #Open original image
                img_to_augment = Image.open(source_path)

                for aug in augs:
                    #augment image
                    augmented_img = augment_image(img_to_augment, aug, device, models[aug], img_size)
                    
                    with open(f'{destination_path}_{aug}.jpg', 'wb') as file:
                        #Save the flipped image
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

#python augment_data.py --original_train_dir data/train_set -a mosaic
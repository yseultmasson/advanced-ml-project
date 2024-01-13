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


def style_transfer(image, style, device):
    image = transforms(image).unsqueeze(0).to(device)

    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),          # scale shortest side to image_size
        transforms.CenterCrop(args.image_size),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)      # normalize with ImageNet values
    ])

    return image


def augment_image(image, augmentation, device):

    if augmentation == 'flip':
        augmented_img = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    elif augmentation == 'starry_night':
        augmented_image = style_transfer(image, augmentation, device)

    else:
        raise ValueError("Invalid augmentation value. It must be either 'flip' or 'style_1'.")

    return augmented_img


def main(args):

    #Set the arguments
    original_train_path = args.original_train_dir
    augmentation = args.augmentation

    

    if 'starry_night' in augmentation:
        # useful constants
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        #Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        style_model = ImageTransformNet().to(device)
        style_model.load_state_dict(torch.load(args.model_path))


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

            #Augment the image and save it
            with open(f'{destination_path}_{augmentation}.jpg', 'wb') as file:
                try:
                    #Open original image
                    img_to_augment = Image.open(source_path)
                    #Flip horizontal
                    augmented_img = augment_image(img_to_augment, augmentation, device)
                    #Save the flipped image
                    augmented_img.save(file)
                except Exception as e:
                    print(e)


            
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('original_train_dir', help='Not augmented train set path')
    parser.add_argument('-a', '--augmentation', help='Augmentation type', type=str)
    
    args = parser.parse_args()

    main(args)
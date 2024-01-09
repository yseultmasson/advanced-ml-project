from PIL import Image
import os
import shutil
from argparse import ArgumentParser
from tqdm import tqdm

def augment_image(image, augmentation):

    if augmentation == 'flip':
        augmented_img = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    elif augmentation == 'style_1':
        print("Not done yet")

    else:
        raise ValueError("Invalid augmentation value. It must be either 'flip' or 'style_1'.")

    return augmented_img


def main(args):

    #Set the arguments
    original_train_path = args.original_train_dir
    augmentation = args.augmentation

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
            destination_path = os.path.join(train_path, category_folder, image)

            #Copy the original image
            shutil.copyfile(source_path, destination_path)

            #Augment the image and save it 
            with open(f'{destination_path}_{augmentation}.png', 'wb') as file:
                #Open original image
                img_to_augment = Image.open(source_path)
                #Flip horizontal
                augmented_img = augment_image(img_to_augment, augmentation)
                #Save the flipped image
                augmented_img.save(file, 'PNG')


            
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('original_train_dir', help='Not augmented train set path')
    parser.add_argument('-a', '--augmentation', help='Augmentation type')
    
    args = parser.parse_args()

    main(args)
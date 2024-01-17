#This script is used to create the train, validation and test sets that we will use for the classification task.

import os
import shutil
from sklearn.model_selection import train_test_split

# Set the path to the Cattech101 dataset
dataset_path = 'data/101_ObjectCategories'

# Set the path to the train, validation and test directories
train_path = 'data/train_set'
test_path = 'data/test_set'
val_path = 'data/val_set'

for path in [train_path, test_path, val_path]:
    os.mkdir(path)

# Iterate through each category folder
for category_folder in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category_folder)

    # Create folders for train, validation and test sets
    for path in [train_path, test_path, val_path]:
        os.mkdir(os.path.join(path, category_folder))

    # List all images in the category folder
    images = os.listdir(category_path)

    # Split the images into train, validation and test sets
    train_images, test_images = train_test_split(images, test_size=0.3, random_state=42)
    train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)

    # Move images to the train folder
    for image in train_images:
        source_path = os.path.join(category_path, image)
        destination_path = os.path.join(train_path, category_folder, image)
        shutil.copyfile(source_path, destination_path)

    # Move images to the test folder
    for image in test_images:
        source_path = os.path.join(category_path, image)
        destination_path = os.path.join(test_path, category_folder, image)
        shutil.copyfile(source_path, destination_path)

    # Move images to the val folder
    for image in val_images:
        source_path = os.path.join(category_path, image)
        destination_path = os.path.join(val_path, category_folder, image)
        shutil.copyfile(source_path, destination_path)

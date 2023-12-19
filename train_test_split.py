import os
import shutil
from sklearn.model_selection import train_test_split

# Set the path to your dataset
dataset_path = '../256_ObjectCategories/256_ObjectCategories'

# Set the path to the train and test directories
train_path = '../256_ObjectCategories/train_set'
test_path = '../256_ObjectCategories/test_set'

# Iterate through each category folder
for category_folder in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category_folder)

    os.mkdir(os.path.join(train_path, category_folder))
    os.mkdir(os.path.join(test_path, category_folder))

    # List all images in the category folder
    images = os.listdir(category_path)

    # Split the images into train and test sets
    train_images, test_images = train_test_split(images, test_size=0.3, random_state=42)

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

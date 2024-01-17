# This script is used to fine-tune a VGG16 network on an image dataset, with specified data augmentation. The train set must have been augmented beforehand, see augment_data.py.

# Example of use, with no augmentation:
# python classification.py --train_dir data/train_set --val_dir data/val_set -a no_aug
# With "mosaic" augmentation
# python classification.py --train_dir data/train_set_mosaic --val_dir data/val_set -a mosaic
# With "mosaic" and "flip" augmentation :
# python classification.py --train_dir data/train_set_mosaic-flip --val_dir data/val_set -a mosaic_flip


import os
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import csv


def train_val_sets(train_dir: str, val_dir: str, batch_size: int, img_height: int, img_width: int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create tensorflow training and validation datasets from image directories.

    Parameters:
    - train_dir (str): Path to the training dataset directory.
    - val_dir (str): Path to the validation dataset directory.
    - batch_size (int): Batch size for training and validation datasets.
    - img_height (int): Height of the input images.
    - img_width (int): Width of the input images.

    Returns:
    - The training dataset and the validation dataset.
    """

    # Training set
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="categorical")

    # Validation set
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="categorical")

    return train_ds, val_ds



def preprocessing(set: tf.data.Dataset) -> tf.data.Dataset:
    """Perform preprocessing steps on a given dataset for VGG16 model."""

    # Preprocessing for VGG16
    set = set.map(lambda x, y: (preprocess_input(x), y))

    # Prefetch data to overlap data preprocessing and model execution during training
    set = set.prefetch(tf.data.AUTOTUNE)

    return set



def custom_vgg16(img_height: int, img_width: int, num_classes: int, weights: str = 'imagenet') -> Model:
    """
    Create a custom VGG16 model with additional fully connected layers.

    Parameters:
    - img_height (int): Height of the input images.
    - img_width (int): Width of the input images.
    - num_classes (int): Number of output classes.
    - weights (str): Specify whether to use 'imagenet' weights or None. Default is 'imagenet'.

    Returns:
    - tensorflow.keras.models.Model: Custom VGG16 model.
    """
    # The base model is a pretrained VGG16 network
    base_model = VGG16(weights=weights, include_top=False, input_shape=(img_height, img_width, 3))

    # Add custom fully connected layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Final model
    return Model(inputs=base_model.input, outputs=predictions)


def callbacks(checkpoint_path: str, history_path: str, patience: int) -> list:
    """
    Define callbacks for model training.

    Parameters:
    - checkpoint_path (str): Path to save the model weights.
    - history_path (str): Path to store the results of each epoch.
    - patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
    - List of callback objects.
    """

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

    # Define checkpoint callback to save the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                    save_weights_only=True,
                                    verbose=1)
    
    # Store results of each epoch in a CSV file
    csv_logger = CSVLogger(history_path, append=True)
    
    return [early_stopping, cp_callback, csv_logger]



def fine_tune(model, train_ds, val_ds, augmentation: str, epochs: int, learning_rate: float) -> None:
    """
    Fine-tune the custom VGG16 model.

    Parameters:
    - model: The model to be fine-tuned.
    - train_ds: Training dataset.
    - val_ds: Validation dataset.
    - augmentation (str): String indicating the types of augmentation applied.
    - epochs (int): Maximum number of training epochs.
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - None
    """

    # Define checkpoint callback to save the model's weights
    checkpoint_path = f'checkpoints/{augmentation}'
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    history_path = f"histories/model_history_{augmentation}.csv"

    # Freeze pretrained layers
    for layer in model.layers[:19]:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Measure initial train and validation scores
    train_loss, train_acc = model.evaluate(train_ds, verbose=1)
    print("Initial train scores measured")
    val_loss, val_acc = model.evaluate(val_ds, verbose=1)
    print("Initial val scores measured")

    # Store initial scores (before training)
    with open(history_path, 'w', newline='') as hist:
        filewriter = csv.writer(hist, delimiter=',')
        filewriter.writerow(['epoch','accuracy','loss','val_accuracy','val_loss'])
        filewriter.writerow([-1, train_acc, train_loss, val_acc, val_loss])

    # Train the model (custom layers only)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs//2,  # Set a maximum number of epochs
        callbacks=callbacks(f'{checkpoint_path}/first_training.ckpt', history_path, patience=3)
    )


    # Fine-tuning of convolutional layers. 
    # Train the top VGG16 block, i.e. freeze the first 15 layers and train the remaining top layers.
    for layer in model.layers[:15]:
        layer.trainable = False
    for layer in model.layers[15:]:
        layer.trainable = True

    # Recompile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate/10), loss='categorical_crossentropy', metrics=['accuracy'])

    # Scores before second training
    train_loss, train_acc = model.evaluate(train_ds, verbose=1)
    val_loss, val_acc = model.evaluate(val_ds, verbose=1)
    with open(history_path,'a') as hist: 
        filewriter = csv.writer(hist, delimiter=',')
        filewriter.writerow([-1, train_acc, train_loss, val_acc, val_loss])
    
    # Train the model (top VGG16 block + custom layers)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs//2,  # Set a maximum number of epochs
        callbacks=callbacks(f'{checkpoint_path}/second_training.ckpt', history_path, patience=5)
    )



def main(args):

    train_dir = args.train_dir
    val_dir = args.val_dir
    batch_size = args.batch_size
    img_height = args.img_height
    img_width = args.img_width
    epochs = args.epochs
    augmentation = args.augmentation
    learning_rate = args.learning_rate

    # Train and validation sets
    train_ds, val_ds = train_val_sets(train_dir, val_dir, batch_size, img_height, img_width)
    print("Datasets loaded")

    # Number of classes
    num_classes = len(train_ds.class_names)

    # Preprocessing
    train_ds = preprocessing(train_ds)
    val_ds = preprocessing(val_ds)
    print("Datasets preprocessed")

    # Create model
    model = custom_vgg16(img_height, img_width, num_classes)

    # Fine-tune the model
    fine_tune(model, train_ds, val_ds, augmentation, epochs, learning_rate)



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train_dir', help='Train set path', required=True)
    parser.add_argument('--val_dir', help='Validation set path', required=True)
    parser.add_argument('-a', '--augmentation', help='Augmentation type', required=True)
    parser.add_argument('-b', '--batch_size', help='Batch size', default=32)
    parser.add_argument('-e', '--epochs', help='Number of epochs', default=40, type=int)
    parser.add_argument('--img_height', help='Image height', default=180)
    parser.add_argument('--img_width', help='Image width', default=180)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate for the first training (divided by 10 for second training)', default=0.0001, type=float)
    
    args = parser.parse_args()

    #Launch code
    main(args)
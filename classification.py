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


def train_val_sets(data_dir, batch_size, img_height, img_width):
    #Training set
    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical")

    #Validation set
    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical")

    return train_ds, val_ds



def preprocessing(set):
    #Preprocessing for vgg16
    set = set.map(lambda x, y: (preprocess_input(x), y))

    #prefetch overlaps data preprocessing and model execution while training.
    set = set.prefetch(tf.data.AUTOTUNE)

    return set


def custom_vgg16(img_height, img_width, num_classes, weights='imagenet'):
    #The base model is a pretrained VGG16 network
    base_model = VGG16(weights=weights, include_top=False, input_shape=(img_height, img_width, 3))

    #Add custom fully connected layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    #Final model
    return Model(inputs=base_model.input, outputs=predictions)


def callbacks(checkpoint_path, history_path):
    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    #Define checkpoint callback to save the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                    save_weights_only=True,
                                    verbose=1)
    
    #Store results of each epoch
    csv_logger = CSVLogger(history_path, append=True)
    
    return [early_stopping, cp_callback, csv_logger]


def fine_tune(model, train_ds, val_ds, augmentation, epochs, learning_rate):
    #Define checkpoint callback to save the model's weights
    checkpoint_path_1 = f"training_{augmentation}/ckpt_top_layers.ckpt"
    checkpoint_path_2 = f"training_{augmentation}/ckpt.ckpt"

    history_path = f"histories/model_history_{augmentation}_log.csv"

    #Freeze pretrained layers
    for layer in model.layers[:19]:
        layer.trainable = False

    #Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    #Store initial scores (before training)
    with open(history_path, 'w', newline='') as hist:
        filewriter = csv.writer(hist, delimiter=',')
        filewriter.writerow(['epoch','accuracy','loss','val_accuracy','val_loss'])
        train_loss, train_acc = model.evaluate(train_ds, verbose=2)
        val_loss, val_acc = model.evaluate(val_ds, verbose=2)
        filewriter.writerow([-1, train_acc, train_loss, val_acc, val_loss])

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs//2,  # Set a maximum number of epochs
        callbacks=callbacks(checkpoint_path_1, history_path)
    )

    np.save(f'histories/first_training_{augmentation}.npy', history.history)

    # Fine-tuning of convolutional layers. 
    #We will freeze the bottom N layers and train the remaining top layers.

    # we chose to train the top vgg16 block, i.e. we will freeze
    # the first 15 layers and unfreeze the rest:
    for layer in model.layers[:15]:
        layer.trainable = False
    for layer in model.layers[15:]:
        layer.trainable = True

    #Recompile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate/10), loss='categorical_crossentropy', metrics=['accuracy'])

    #Scores before second training
    with open(history_path,'a') as hist:
        train_loss, train_acc = model.evaluate(train_ds, verbose=2)
        val_loss, val_acc = model.evaluate(val_ds, verbose=2)
        #hist.write(f"-1, {train_acc}, {train_loss}, {val_acc}, {val_loss}")
        filewriter = csv.writer(hist, delimiter=',')
        filewriter.writerow([-1, train_acc, train_loss, val_acc, val_loss])
    
    # Train the model with early stopping callback
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs//2,  # Set a maximum number of epochs
        callbacks=callbacks(checkpoint_path_2, history_path)
    )
    np.save(f'histories/second_training_{augmentation}.npy',history.history)



def main(args):

    data_dir = args.data_dir
    batch_size = args.batch_size
    img_height = args.img_height
    img_width = args.img_width
    epochs = args.epochs
    augmentation = args.augmentation
    learning_rate = args.learning_rate

    #Train and validation sets
    train_ds, val_ds = train_val_sets(data_dir, batch_size, img_height, img_width)

    #List of the 257 class names
    class_names = train_ds.class_names
    num_classes = len(class_names)

    #Preprocessing
    train_ds = preprocessing(train_ds)
    val_ds = preprocessing(val_ds)

    model = custom_vgg16(img_height, img_width, num_classes)
    fine_tune(model, train_ds, val_ds, augmentation, epochs, learning_rate)



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('data_dir', help='Data path')
    parser.add_argument('-b', '--batch_size', help='Batch size', default=32)
    parser.add_argument('-e', '--epochs', help='Number of epochs', default=40, type=int)
    parser.add_argument('--img_height', help='Image height', default=180)
    parser.add_argument('--img_width', help='Image width', default=180)
    parser.add_argument('-a', '--augmentation', help='Augmentation type', default='no_aug')
    parser.add_argument('-lr', '--learning_rate', help='Learning rate for the first training (divided by 10 for second training)', default=0.0001, type=float)
    
    args = parser.parse_args()

    #Launch code
    main(args)
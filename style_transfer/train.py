"""Main script for model training of the generative approach."""
import torch
import os
from argparse import ArgumentParser
from tqdm import tqdm
from datetime import datetime

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import *
from image_transformer_net import ImageTransformNet
from vgg16 import Vgg16

def train(args:ArgumentParser) -> None:
    """
    Trains the model associated to a style image, and stores it as a .model for later usage.

    Parameters
    ----------
    args : Argument Parser
        arguments passed through a terminal. Here is the list of arguments:
            
        args.style_image : a str. The path of the style image to use for the training.
        args.dataset : a str. The path to the training dataset.
        args.image-size : an int. The size of the input images (both width and height).
        args.batch-size : an int. The batch size used during training.
        args.lr : a float. The learning rate used during optimization.
        args.epochs : an int. The number of training epochs.
        args.style-weight : a float. The weight given to the style loss. Default: 1e5
        args.content-weight : a float. The weight given to the content (feature reconstruction) loss. Default: 1e0
        args.tv-reg : a boolean. If true, adds a total variation regularization term to the total loss.
        args.tv-weight : a float. The weight given to the total variation loss. Default : 1e-7
        args.content-extraction :  an int, between 0 and 3. Layer used to compute the content loss among ('conv1_2', 'conv2_2', 'conv3_3', 'conv4_3')

    Returns
    -------
    None. The function automatically saves the trained model in a .model file.

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate the image transformation network
    image_transformer = ImageTransformNet().to(device) # loads the image transformation network and sends it to the device.
    optimizer = Adam(image_transformer.parameters(), args.lr) # instantiate the optimizer with the desired learning rate.
    loss_mse = torch.nn.MSELoss()

    # instantiate the pretrained vgg network (loss network) and send it to the device
    vgg = Vgg16().to(device)

    # define image preprocessing steps for training data
    dataset_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), # forces the image into a square of size (image_size, image_size)
        transforms.CenterCrop(args.image_size),      # crops the image at the center
        transforms.ToTensor(),                  # convert the image to a [0., 1.] tensor
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)      # normalize the tensor with ImageNet values
    ])

    # load the dataset from the desired folder and pass it to the data loader
    start = datetime.now()
    print("Fetching train data")
    train_dataset = datasets.ImageFolder(args.dataset,
                                         dataset_transform)    
    print(f"Done : {datetime.now() - start}")

    train_loader = DataLoader(train_dataset,
                              batch_size = args.batch_size)


    # define style image preprocessing steps
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)
    ])
    style = load_image(args.style_image)
    style = style_transform(style)
    
    style = Variable(style.repeat(args.batch_size, 1, 1, 1)).to(device) # duplicates the tensor to match the batch size.
    style_name = os.path.split(args.style_image)[-1].split('.')[0] # will be used to create the name of the saved model.
    

    # compute the gram matrices of the style feature maps we care about
    style_features = vgg(style)
    style_gram = [gram(fmap) for fmap in style_features] # gram comes from utils.py

    for e in range(args.epochs):
        print(f"Starting epoch {e + 1}")

        # keep track of the number of processed images as well as the different loss terms
        img_count = 0
        agg_style_loss = 0.0
        agg_content_loss = 0.0
        agg_tv_loss = 0.0

        # train network
        image_transformer.train() # start training the image transformation network.
        for batch_idx, (x, label) in enumerate(tqdm(train_loader, desc=f"Epoch {e + 1}")):
            img_batch_read = len(x)
            img_count += img_batch_read

            # reset the gradient computations
            optimizer.zero_grad()

            # feed the input batch to the image transformation network
            x = Variable(x).to(device)
            y_hat = image_transformer(x)

            # compute the vgg features of the generated image(s) and the content image(s)
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # compute the style reconstruction loss
            y_hat_gram = [gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = args.style_weight * style_loss
            agg_style_loss += style_loss.item()

            # compute the content (feature reconstruction) loss
            recon = y_c_features[args.content_extraction]      
            recon_hat = y_hat_features[args.content_extraction]
            content_loss = args.content_weight * loss_mse(recon_hat, recon)
            agg_content_loss += content_loss.item()

            # compute intermediary total loss
            total_loss = style_loss + content_loss

            # if wanted, compute the total variation regularization and add it to the total loss
            if args.tv_reg:
                diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
                diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
                tv_loss = args.tv_weight * (diff_i + diff_j)
                total_loss += tv_loss
                agg_tv_loss += tv_loss.item()

            # update model weights
            total_loss.backward()
            optimizer.step()

            # display training status every 1000 batches
            if ((batch_idx + 1) % 1000 == 0):
                status = f"""
                Epoch {e + 1}: {img_count}/{len(train_dataset)}
                agg_style_loss: {agg_style_loss / (batch_idx + 1)}
                agg_content_loss: {agg_content_loss / (batch_idx + 1)}
                agg_tv_loss: {agg_tv_loss / (batch_idx + 1)}"""
                print(status)

    # save the trained model
    image_transformer.eval()

    if not os.path.exists("models"):
        os.makedirs("models")
    fn = f"models/{style_name}_{args.epochs}_epochs_{len(train_dataset)}_samples_{args.content_extraction + 1}_{args.content_weight}_cttwght.model"
    torch.save(image_transformer.state_dict(), fn)


if __name__ == '__main__':
    parser = ArgumentParser(description='Train the architecture on a specific style.')

    parser.add_argument(
        "--style-image",
        type=str,
        required=True,
        help="Path of the style image to use for the training.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the training dataset.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Size of the input images (both width and height).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size used during training.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate used during optimization.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.")
    parser.add_argument(
        "--style-weight",
        type=float,
        default=1e5,
        help="Weight given to the style loss.")
    parser.add_argument(
        "--content-weight",
        type=float,
        default=1e0,
        help="Weight given to the content loss.")
    parser.add_argument(
        "--tv-reg",
        action='store_true',
        help="If True, adds a 'total variation' regularization term to the loss.")
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=1e-7,
        help="Weight given to the total variation loss.")
    parser.add_argument(
        "--content-extraction",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Use feature maps after block (content-extraction + 1) for content loss.")

    args = parser.parse_args()
    train(args)

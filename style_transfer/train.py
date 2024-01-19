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
    Loads and train the model associated to a style image, then stores it as a .model to be reused whenever needed

    Parameters
    ----------
    args : Argument Parser
        arguments passed through a terminal. Here is the list of arguments:
            
        args.style_image : a str. The path of the style image to use for the training.
        args.dataset : a str. The path to the training dataset.
        args.image-size : an int. The size of the input images (both width and height).
        args.batch-size : an int. The batch size used during training.
        args.lr : a float. The Learning rate used during optimization.
        args.epochs : an int. The number of training epochs.
        args.style-weight : an int. The weight given to the style loss. Default: 100_000
        args.content-weight : an int. The weight given to the content loss. Default: 1
        
        args.tv-reg : a boolean. If true, adds a total variation regularization term to the total loss.
        args.tv-weight : a float . The weight given to the total variation loss. Default : 1e-7
        args.content-extraction :  an int, between 0 and 3. Every feature map after the block (content-extraction + 1 ) will be used for the content loss.

    Returns
    -------
    None. The function directly pastes the trained model in a file, and stores it in a specified path.

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define network
    image_transformer = ImageTransformNet().to(device) #loads the model to the device
    optimizer = Adam(image_transformer.parameters(), args.lr) # gives an optimizer to the model
    loss_mse = torch.nn.MSELoss() #defines the loss of the model

    # load vgg network
    vgg = Vgg16().to(device)

    # get training dataset
    dataset_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),          # scale shortest side to image_size. The dimension of the image is not conserved.
        transforms.CenterCrop(args.image_size),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)      # normalize with ImageNet values, available in utils.py
    ])

    start = datetime.now() # will be used to calculate the time taken for the loading of the training dataset.
    print("Fetching train data")
    train_dataset = datasets.ImageFolder(args.dataset,
                                         dataset_transform) #loads the dataset    
    print(f"Done : {datetime.now() - start}")

    train_loader = DataLoader(train_dataset,
                              batch_size = args.batch_size)


    # style image. It is neither cropped, nor resized.
    style_transform = transforms.Compose([
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)      # normalize with ImageNet values
    ])
    style = load_image(args.style_image)
    style = style_transform(style)
    
    style = Variable(style.repeat(args.batch_size, 1, 1, 1)).to(device) #repeats the tensor data to match the batch size.
    style_name = os.path.split(args.style_image)[-1].split('.')[0] # will be used to create the name of the saved model.
    

    # calculate gram matrices for style feature layer maps we care about
    style_features = vgg(style)
    style_gram = [gram(fmap) for fmap in style_features] #gram comes from utils.py

    for e in range(args.epochs):
        print(f"Starting epoch {e + 1}")

        # track values for...
        img_count = 0
        agg_style_loss = 0.0
        agg_content_loss = 0.0
        agg_tv_loss = 0.0

        # train network
        image_transformer.train() #informs the code that the model is being trained ==> the instancenorm2d updates from the ImageTransformNet class are frozen.
        for batch_idx, (x, label) in enumerate(tqdm(train_loader, desc=f"Epoch {e + 1}")):
            img_batch_read = len(x)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            x = Variable(x).to(device)
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # calculate style loss
            y_hat_gram = [gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = args.style_weight * style_loss
            agg_style_loss += style_loss.item()

            # calculate content loss
            recon = y_c_features[args.content_extraction]      
            recon_hat = y_hat_features[args.content_extraction]
            content_loss = args.content_weight * loss_mse(recon_hat, recon)
            agg_content_loss += content_loss.item()

            # total loss
            total_loss = style_loss + content_loss

            # calculate total variation regularization (anisotropic version). For more details, see the link below:
            # https://www.wikiwand.com/en/Total_variation_denoising
            if args.tv_reg:
                diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
                diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
                tv_loss = args.tv_weight * (diff_i + diff_j)
                total_loss += tv_loss
                agg_tv_loss += tv_loss.item()

            # backwards propagation
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_idx + 1) % 1000 == 0):
                status = f"""
                Epoch {e + 1}: {img_count}/{len(train_dataset)}
                agg_style_loss: {agg_style_loss / (batch_idx + 1)}
                agg_content_loss: {agg_content_loss / (batch_idx + 1)}
                agg_tv_loss: {agg_tv_loss / (batch_idx + 1)}"""
                print(status)

    # save model
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

"Main script to perform descriptive style transfer."
import torch
import os
from argparse import ArgumentParser
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from utils import *
from vgg16 import Vgg16

def compute_style_loss(loss_fn, gram_style, gram_new_image, weights):
    style_loss = 0.0

    for j in range(len(gram_style)):
        style_loss += loss_fn(gram_new_image[j], gram_style[j]) # / (weights[j] ** 2) # / (gram_style[j].shape[2]) ** 2

    return style_loss # / (4 * len(gram_style))


def compute_content_loss(loss_fn, content_features, new_image_features):
    return loss_fn(content_features, new_image_features) / 2


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Vgg16(mode="descriptive_st").to(device)

    content_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)
    ])

    content = load_image(args.content_image)
    content = content_transform(content).to(device) #content.size() = torch.Size([3, 256, 256])
    content_features = model(content)
    content_feats = content_features[-1].unsqueeze(0)
    # content_gram = [gram(fmap) for fmap in content_features]

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)
    ])

    style_name = os.path.split(args.style_image)[-1].split('.')[0]
    style = load_image(args.style_image)
    style = style_transform(style).unsqueeze(0).to(device)
    style_features = model(style)
    style_gram = [gram(fmap) for fmap in style_features[:-1]] #pour style loss comme pour content loss, on ne calcule que les features qui nous intéressent.

    new_image = torch.rand((3, args.image_size, args.image_size)).unsqueeze(0).to(device)
    new_image = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(new_image)
    new_image.requires_grad_(True)

    optimizer = Adam([new_image], args.lr)
    loss_mse = torch.nn.MSELoss(reduction='sum')

    for i in tqdm(range(args.iterations), desc="Performing style transfer"):

        optimizer.zero_grad()
        img_features = model(new_image)
        wghts = [(f.shape[1] * f.shape[2] * f.shape[3]) for f in img_features[:-1]]

        img_gram = [gram(fmap) for fmap in img_features[:-1]]
        content_loss = compute_content_loss(loss_fn=loss_mse,
                                            content_features=content_feats,
                                            new_image_features=img_features[-1])
        style_loss = compute_style_loss(loss_fn=loss_mse,
                                        gram_style=style_gram,
                                        gram_new_image=img_gram,
                                        weights=wghts)

        loss = args.content_weight * content_loss + args.style_weight * style_loss
        
        loss.backward()
        optimizer.step()
        step_scheduler.step() #sans utiliser de step_scheduler, la content loss finit par remonter mdr tout va bien
        
        if (i + 1) % 200 == 0:
            print(f"Iteration {i}/{args.iterations} - Style loss : {style_loss} - Content loss : {content_loss}")
            filename = f"data/results_dst/{style_name}_style_iteration_{i + 1}.jpg"
            cpu_img = new_image.cpu() 
            save_image(filename, cpu_img.data[0])
            print(len(img_gram))


if __name__ == '__main__':
    parser = ArgumentParser(description='Train the architecture on a specific style.')

    parser.add_argument(
        "--style-image",
        type=str,
        required=True,
        help="Path of the style image to use for the training.")
    parser.add_argument(
        "--content-image",
        type=str,
        required=True,
        help="Path to the source image to stylize.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Size of the input images (both width and height).")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-1,
        help="Learning rate used during optimization.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=4600,
        help="Number of iterations.")
    parser.add_argument(
        "--style-weight",
        type=float,
        default=1000,
        help="Weight given to the style loss.")
    parser.add_argument(
        "--content-weight",
        type=float,
        default=1e0,
        help="Weight given to the content loss.")

    args = parser.parse_args()
    main(args)




"""Style transfer."""
from argparse import ArgumentParser
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import torch
import os
import re

from utils import *
from image_transformer_net import ImageTransformNet

def preprocess_image(img_filename:str, transforms:transforms.Compose, device:torch.device):
    """
    Loads the image, and sends it to the device used for the generation (GPU if possible, else CPU).

    Parameters
    ----------
    img_filename : str
        the relative path leading to the image.
    transforms : transforms.Compose
        the transformations done to the image. It is passed as an argument for flexibility.
    device : torch.device
        the device used for the generation (GPU if possible, else CPU).

    Returns
    -------
    tensor : 
        the converted image.

    """
    ctt = load_image(img_filename)
    ctt = transforms(ctt).unsqueeze(0)

    return ctt.to(device)


def style_transfer(args:ArgumentParser) -> None :
    """
    Uses an already trained model to perform style transfer over some base images.

    Parameters
    ----------
    args : ArgumentParser
        arguments passed through a terminal. Here is the list of arguments:
            
            args.model-path : a str. The path leading to the trained model to use.
            args.source: a str. The path to the folder containing the images to stylize. Many images can be transformed at once.
            args.output: a str. The path where the stylized images will be saved.
            args.image-size: a float. The size (both height and width: if the image is not squared, it will be after the transform) of the output image

    Returns
    -------
    None. Everything is done inside the function.

    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # content image
    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), # forces the image into a square of size (image_size, image_size)
        transforms.CenterCrop(args.image_size),      # crops the image at the center
        transforms.ToTensor(),                  # convert the image to a [0., 1.] tensor
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)      # normalize the tensor with ImageNet values
    ])

    style_model = ImageTransformNet().to(device) # loads the image transformation network and sends it to the device.
    style_model.load_state_dict(torch.load(args.model_path)) # loads the weights of the desired model.

    pattern = re.compile(r"\/([a-zA-Z_]+)_\d+_epochs_\d+_samples_\d+_\d+\.\d+_cttwght\.model")
    model_name = pattern.search(args.model_path).group(1)
    output_dir = os.path.join(args.output, model_name)

    if not os.path.exists(output_dir): # checks if the output directory exists. If not, creates it.
        os.makedirs(output_dir)

    start = datetime.now()
    count = 0
    for img_fn in tqdm(os.listdir(args.source), desc="Stylizing images"): # stylize all images present in the source directory.
        img_path = os.path.join(args.source, img_fn)
        content = preprocess_image(img_path, image_transform, device) # preproces the image before feeding it to the model.
        stylized = style_model(content).cpu() # stylize the image and bring it back onto the cpu before saving it.
        out_im_fn = f"{model_name}_{img_fn}"
        save_image(os.path.join(output_dir, out_im_fn), stylized.data[0])
        count += 1

    print(f"Average inference time : {(datetime.now() - start) / count}")

if __name__ == '__main__':
    parser = ArgumentParser(description='Apply a style to an image.')

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the desired style model.")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the folder containing images to transform.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where to save the stylized images.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Size of the input images (both width and height).")

    args = parser.parse_args()
    style_transfer(args)

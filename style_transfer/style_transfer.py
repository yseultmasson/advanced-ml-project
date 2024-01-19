"""Style transfer."""
from argparse import ArgumentParser
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import torch
import os
import re

from utils import *
from image_transformer_net import ImageTransformNet

def preprocess_image(img_filename, transforms, device):
    ctt = load_image(img_filename)
    ctt = transforms(ctt).unsqueeze(0)

    return ctt.to(device) # Variable(ctt).to(device)


def style_transfer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # content image
    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),          # scale shortest side to image_size
        transforms.CenterCrop(args.image_size),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)      # normalize with ImageNet values
    ])

    # content = load_image(args.source)
    # content = image_transform(content)
    # content = content.unsqueeze(0)
    # content = Variable(content).to(device)

    style_model = ImageTransformNet().to(device)
    style_model.load_state_dict(torch.load(args.model_path))

    pattern = re.compile(r"\/([a-zA-Z_]+)_\d+_epochs_\d+_samples_\d+_\d+\.\d+_cttwght\.model")
    model_name = pattern.search(args.model_path).group(1)
    output_dir = os.path.join(args.output, model_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = datetime.now()
    count = 0
    for img_fn in tqdm(os.listdir(args.source), desc="Stylizing images"):
        img_path = os.path.join(args.source, img_fn)
        content = preprocess_image(img_path, image_transform, device)
        stylized = style_model(content).cpu()
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

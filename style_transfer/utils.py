"""Useful constants and functions."""
from PIL import Image
import numpy as np
import torch

# useful constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# opens and returns image file as a PIL image (0-255)
def load_image(filename:str):
    """
    Opens and returns an image from a filename. It is returned as a PIL image, whose RGB values range from 0 to 255.

    Parameters
    ----------
    filename : str
        The name of the input file.

    Returns
    -------
    img : Image

    """
    img = Image.open(filename)
    return img

def save_image(filename:str, data:Image) -> None:
    """
    saves an image, assuming its data comes in batch form (channels, height, width)

    Parameters
    ----------
    filename : str
        The name of the output file.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None. Everything is done inside the function.

    """
    mean = np.array(IMAGENET_MEAN).reshape((3, 1, 1))
    std = np.array(IMAGENET_STD).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

# Calculate Gram matrix (G = FF^T)
def gram(x:torch.Tensor) -> np.float32:
    """
    Calculates the gram matrix of a vector

    Parameters
    ----------
    x : a Tensor

    Returns
    -------
    G : np.float32
        The Gram matrix associated to x.

    """
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

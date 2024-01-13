# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 07:10:23 2024

@author: rayan
"""

# -*- coding: utf-8 -*-

    
#%%
# from tqdm import tqdm
# import time

# file=open('progress.txt','w')
# items=50
# for i in tqdm(range(items),file=file):
#     time.sleep(0.1)
# file.close()   

#%%

from PIL import Image
import numpy as np

# useful constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# opens and returns image file as a PIL image (0-255)
def load_image(filename):
    img = Image.open(filename)
    return img

# assumes data comes in batch form (ch, h, w)
def save_image(filename, data):
    mean = np.array(IMAGENET_MEAN).reshape((3, 1, 1))
    std = np.array(IMAGENET_STD).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

# Calculate Gram matrix (G = FF^T à vérifier que ce n'est F^T*F)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

#%%
import torch
import os
from argparse import ArgumentParser
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm

from vgg16 import Vgg16

style_image=r"C:\Users\rayan\Documents\GitHub\advanced-ml-project\images\style\starry_night.jpg"
content_image=r"C:\Users\rayan\Documents\GitHub\advanced-ml-project\images\base_images\bus.jpg"
image_size=256

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Vgg16(mode="descriptive_st").to(device)
    content_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        ])
    content = load_image(content_image)
    content = content_transform(content).to(device)
    content_features = model(content) #Me sort 6 tensors correspondants aux 6 couches que vgg16 me retourne. content features [-1] = les features de l'image de base dans la dernière couche du réseau de neurones.
    # content_gram = [gram(fmap) for fmap in content_features] Pas besoin des matrices de Gram
    
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)])
    
    style = load_image(style_image)
    style = style_transform(style).unsqueeze(0).to(device)
    style_features = model(style) #Me sort 6 tensors correspondants aux 6 couches que vgg16 me retourne.
    style_gram = [gram(fmap) for fmap in style_features[:-1]] #pour style loss comme pour content loss, on ne calcule que les features et les matrices de gram qui nous intéressent.
    
    new_image = torch.rand((3, image_size, image_size)).unsqueeze(0).to(device)
    new_image.requires_grad_(True)
    
    img_features = model(new_image)
    img_gram = [gram(fmap) for fmap in img_features[:-1]]
    
#ça devrait me ressortir une image très proche du tableau de Van Gogh, non?
    transforms.ToPILImage()(style[0])

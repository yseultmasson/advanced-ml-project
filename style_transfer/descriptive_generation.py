"""
This code is a rework of an available GitHub repository available here : https://github.com/nazianafis/Neural-Style-Transfer . This repository is itself an implementation of the work of Gatys et Al (2016).
Their article is available here: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf


This code is a part of the following repository: https://github.com/yseultmasson/advanced-ml-project
It is supposed to run seamlessly as long as the repository is properly coded. If you want to transfer different styles on different base images, follow the instructions in the section "parameters to run the code" at the end of the code.



"""
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import LBFGS
import os
from vgg16 import Vgg16 #custom version of the VGG16 model built in Pytorch.
import time

# Constants for image normalization
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def load_image(img_path:str ,target_shape="None") -> np.float32 :
    '''
    Loads and resize the image, turning it into a np.array whose values are between 0 and 1.
    '''
    if not os.path.exists(img_path):
        raise Exception(f'Path not found: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]                   # convert BGR to RGB when reading
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32)
    img /= 255.0
    return img

def prepare_img(img_path:str, target_shape:str, device:torch.device) -> torch.Tensor :
    '''
    Normalizes the image.
    '''
    img = load_image(img_path, target_shape=target_shape)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)])
    img = transform(img).to(device).unsqueeze(0)
    return img

def save_image(img:np.float32, img_path:str) -> None :
    """
    Saves the image to the specified img_path.
    """
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    cv.imwrite(img_path, img[:, :, ::-1])                   # convert RGB to BGR while writing

def generate_out_img_name(config:dict) -> str :
    '''
    Generates a name for the output image.
    Example: 'c1-s1.jpg'
    where c1: content_img_name, and
          s1: style_img_name.
          
    config is a dictionary of parameters. See the documentation of the function "neural_style_transfer" for more details.
    '''
    prefix = os.path.basename(config['content_img_name']).split('.')[0] + '_' + os.path.basename(config['style_img_name']).split('.')[0]
    suffix = f'{config["img_format"][1]}'
    return prefix + suffix

def save_and_maybe_display(optimizing_img:torch.Tensor, dump_path:str, config:dict, img_id:float, num_of_iterations:float) -> None:
    '''
    Saves the generated image to a specified dump_path
    config is a dictionary of parameters. See the documentation of the function "neural_style_transfer" for more details.
    If saving_freq == -1, only the final output image will be saved.
    Else, intermediate images can be saved too.
    '''
    saving_freq = -1
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)

    if img_id == num_of_iterations-1 :
        img_format = config['img_format']
        out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else generate_out_img_name(config)
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])
    

def prepare_model(device:torch.device) -> Vgg16:
    '''
    Load Vgg16 model into local cache. See vgg16.py for more details about our implementation of the model.
    '''
    model = Vgg16(mode="descriptive_st")
    return model.to(device).eval()
#paste here
def gram_matrix(x:np.float32, should_normalize=True) -> np.float32:
    '''
    Generate gram matrices of a matrix x. Will be used for the matrices the representations of content and style images.
    '''
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram

def total_variation(y : torch.Tensor) -> float :
    '''
    Calculate the total variation of a tensor.
    '''
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

def build_loss(neural_net:Vgg16, optimizing_img:torch.Tensor, target_representations:list, content_feature_maps_index:float, style_feature_maps_indices:list, config:dict):
    '''
    Calculates the content_loss and the style_loss of a generated image. Its style and content reconstructions are done within the function.
    
    PARAMETERS:
        neural_net: the neural network. We use a custom version of the Vgg16, tailored to our needs.
        optimizing_img : the image we wish to optimize by minimizing the loss functions. 
        target_representations : the content extraction and style extractions of the base image, in that order.
        content_feature_map_index : the index corresponding to the content extraction of our model. In our case, it is -1. See Vgg16.py for more information.
        style_feature_maps_indices : the indices corresponding to the style extractions of our model. In our case, it is [0,1,2,3,4]. See Vgg16.py for more information.
        config : a dictionary of parameters. See the documentation of the function "neural_style_transfer" for more details
    '''
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]
    
    current_set_of_feature_maps = neural_net(optimizing_img)
    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation) #Mean Squared Error Loss.
    
    style_loss = 0.0
    current_style_representation = [gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)
    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss
    return total_loss, content_loss, style_loss, 

def make_tuning_step(neural_net:Vgg16, optimizer , target_representations:list, content_feature_maps_index:float, style_feature_maps_indices:list, config:dict):
    '''
    Performs a step in the tuning loop.
    (We are tuning only the pixels, not the weights.)
    All the parameters are the same than for the function "build_loss", except for "optimizer"
    PARAMETERS:
        neural_net: the neural network. We use a custom version of the Vgg16, tailored to our needs.
        optimizer : the optimizing function that will be used to perform gradient descent. In our case, we use LBFGS, but others will also work.
        target_representations : the content extraction and style extractions of the base image, in that order.
        content_feature_map_index : the index corresponding to the content extraction of our model. In our case, it is -1. See Vgg16.py for more information.
        style_feature_maps_indices : the indices corresponding to the style extractions of our model. In our case, it is [0,1,2,3,4]. See Vgg16.py for more information.
        config : a dictionary of parameters. See the documentation of the function "neural_style_transfer" for more details

    '''
    def tuning_step(optimizing_img):
        """
        the tuning_step itself. Returns the total_loss, the content_loss and the style_loss.

        """
        total_loss, content_loss, style_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config) #was a tv_loss here.
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss
    return tuning_step

def neural_style_transfer(config):
    '''
    The main Neural Style Transfer method.
    config is a dictionary of parameters for the generation. Its keys are: 
        'content_img_name': the name of the base image on which the program will try to apply a style. Only its name, not the path leading to it.
         'style_img_name' : the name of the image whose style will be reproduced. Only its name, not the path leading to it .
         'height' : the height of the output image. The width will automatically be calculated to preserve the ratio of the image.
         'content_weight' : the weight given to the content loss.
         'style_weight' : the weight given to the style loss.
         'content_images_dir' : the path leading to the base image.
         'style_images_dir' : the path leading to the style image. This configuration of parameters allows for an easy automation of multiple generations.
         'output_img_dir' : the path leading to the output image. 
         img_format'
         
    This function returns the path leading to the output images. 
    '''
    
    # storing the paths leading to the style and base images. Creating the directory of the output image if needed.
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    # creates a directory, in which the output image will be stored. While the creation of a directory for one image might seem useless, it can be quite practical when we also want to save some intermediate output images, which is made possible by the function "save_and_maybe_display".
    
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)
    
    #preparing the images. the initial image is the content_image, instead of a white noise image. Both methods areroughly similar according to Gatys et. Al
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = prepare_img(content_img_path, config['height'], device)
    style_img = prepare_img(style_img_path, config['height'], device)
    init_img = content_img
    optimizing_img = Variable(init_img, requires_grad=True)
    neural_net = prepare_model(device)    
    
    
    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)
    
    
    target_content_representation = content_img_set_of_feature_maps[-1].squeeze(axis=0)  # this command line retrieves the feature map of the content representation
    target_style_representation = [gram_matrix(fmap) for cnt,fmap in enumerate(style_img_set_of_feature_maps[:-1])] # this command line retries the gram matrices of the feature maps of the style representation

    
    target_representations = [target_content_representation, target_style_representation]
    num_of_iterations = 1000
    
    optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
    cnt = 0
    start=time.time()
    def closure():
        
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss, content_loss, style_loss = build_loss(neural_net, optimizing_img, target_representations, -1, [0,1,2,3,4], config) 
        if total_loss.requires_grad:
            total_loss.backward()
        with torch.no_grad():
            save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations)
            # if cnt % 100 == 0 :
            #     print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}') #, tv loss={config["tv_weight"] * tv_loss.item():12.4f}
        cnt += 1
        return total_loss
    optimizer.step(closure)
    
    end=time.time()
    elapsed = round(end-start,3)
    print(f"elapsed time (in seconds): {elapsed}" )

    return dump_path



# Parameters to run the code

PATH = r'' #The initial directory. Change if needed. The original directory of the GitHub is already designed to make this code work seamlessly.


default_resource_dir = os.path.join(PATH, 'images') # The program will look for all images in the subdirectory "images" of the initial directory.

content_images_dir = os.path.join(default_resource_dir, 'base_images') # The program will look for base images in the subdirectory "base_images" of the images directory.
style_images_dir = os.path.join(default_resource_dir, 'style') # The program will look for style images in the subdirectory "style_images" of the images directory.
output_img_dir = os.path.join(default_resource_dir, r'output_images\descriptive_generation') # The program will create subdirectories in which storing the images in the subdirectory "output_images\descriptive_generation" of the images directory.
img_format = (4, '.jpg')

CONTENT_IMAGE = r'bus.jpg' # this image is found in \images\base_images. Feel free to add any image you want, and change the name accordingly, if you want to test the code on another image.
STYLE_IMAGE = 'starry_night.jpg' # this image is found in \images\style. Feel free to add any image you want, and change the name accordingly, if you want to test the code on another image.

optimization_config = {'content_img_name': CONTENT_IMAGE, 'style_img_name': STYLE_IMAGE, 'height': 256, 'content_weight': 1.0, 'style_weight': 1000.0}
optimization_config['content_images_dir'] = content_images_dir
optimization_config['style_images_dir'] = style_images_dir
optimization_config['output_img_dir'] = output_img_dir
optimization_config['img_format'] = img_format

# Running the code
results_path = neural_style_transfer(optimization_config)
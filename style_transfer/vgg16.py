"""Custom implementation of the VGG16 network architecture."""
import torch.nn as nn
from torchvision import models

class Vgg16(nn.Module):
    def __init__(self, mode='generative_st'):
        super(Vgg16, self).__init__()
        self.mode = mode
        features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features # load pretrained VGG16 network and collect its features.
        self.feat = features

        self.relu_layers = nn.ModuleList()
        layer_indices = [0, 2, 4, 7, 9, 12, 16, 19, 21, 23, 26] # the indices of the layers we will use
        self.relu_names = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_3', '4_1', '4_2', '4_3', '5_1'] # the names of the layers that we will use.

        for i in range(len(layer_indices) - 1):
            layer_start, layer_end = layer_indices[i], layer_indices[i + 1]
            setattr(self, f'to_relu_{self.relu_names[i]}', nn.Sequential(*features[layer_start:layer_end]))

        if self.mode == 'generative_st':
            self.extracted_fmaps = ['1_2', '2_2', '3_3', '4_3'] # the feature maps used for the generative approach
        else:
            self.extracted_fmaps = ['1_1', '2_1', '3_1', '4_1', '4_2', '5_1'] # the feature maps used for the descriptive approach. Note that 'conv4_2' is the one for content reconstruction. 

        # Freeze model parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        feature_maps = []

        for i in range(len(self.relu_names)):
            layer_idx = self.relu_names[i]
            x = getattr(self, f'to_relu_{layer_idx}')(x)
            if layer_idx in self.extracted_fmaps:
                feature_maps.append(x)
            
            if (self.mode == 'generative_st') and (len(feature_maps) == len(self.extracted_fmaps)):
                return tuple(feature_maps) # return the feature_maps used by the generative approach.
        
        feature_maps[-2], feature_maps[-1] = feature_maps[-1], feature_maps[-2]
        # after this line, the order of the feature maps is as follows for the descriptive approach: ['1_1', '2_1', '3_1', '4_1', '5_1, '4_2'].
        # The first five items correspond to the style extraction, the last one corresponds to content extraction. This order allows to use index -1 elsewhere to access content features.

        return tuple(feature_maps)

"""Loss network (vgg16)."""
import torch.nn as nn
from torchvision import models

class Vgg16(nn.Module):
    def __init__(self, mode='generative_st'):
        super(Vgg16, self).__init__()
        self.mode = mode
        features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.feat = features

        self.relu_layers = nn.ModuleList()
        layer_indices = [0, 2, 4, 7, 9, 12, 16, 19, 21, 23, 26]
        self.relu_names = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_3', '4_1', '4_2', '4_3', '5_1']

        for i in range(len(layer_indices) - 1):
            layer_start, layer_end = layer_indices[i], layer_indices[i + 1]
            setattr(self, f'to_relu_{self.relu_names[i]}', nn.Sequential(*features[layer_start:layer_end]))

        if self.mode == 'generative_st':
            self.extracted_fmaps = ['1_2', '2_2', '3_3', '4_3']
        else:
            self.extracted_fmaps = ['1_1', '2_1', '3_1', '4_1', '4_2', '5_1']

        # Freeze parameters
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
                return tuple(feature_maps)
        
        feature_maps[-2], feature_maps[-1] = feature_maps[-1], feature_maps[-2]
        return tuple(feature_maps)

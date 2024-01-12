"""Loss network (vgg16)."""
import torch.nn as nn
from torchvision import models

class Vgg16(nn.Module):
    def __init__(self, mode='generative_st'):
        super(Vgg16, self).__init__()
        self.mode = mode
        features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.feat = features
        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_4_2 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.to_relu_5_1 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2, 4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 19):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(19, 21):
            self.to_relu_4_2.add_module(str(x), features[x])
        for x in range(21, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        for x in range(23, 26):
            self.to_relu_5_1.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_1_2(h)
        h_relu_1_2 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_4_2(h)
        h_relu_4_2 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h

        if self.mode == 'generative_st':
            return (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        
        h = self.to_relu_5_1(h)
        h_relu_5_1 = h

        return (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_5_1, h_relu_4_2)

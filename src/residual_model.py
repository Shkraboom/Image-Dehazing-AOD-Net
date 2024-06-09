import torch
import torch.nn as nn
import cv2
import numpy as np
import h5py

class ResidualBlock(nn.Module):
    """
    One block for RNN
    """

    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features = in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)
    
    def forward(self, x_input):
        """
        One block's forward pass 

        Input: x_input - input tensor (1, 16, 16, 16)
        Output: x_output - output tensor (1, 16, 16, 16)
        """

        x_shortcut = x_input.clone()

        x = self.batch_norm(x_input)
        x = self.relu(x)
        x = self.conv(x)

        x = x + x_shortcut
        x_output = self.relu(x)

        return x_output
    
class ResidualModel(nn.Module):
    """
    RNN to reconstruct a clean image based on the hazy image and its transmission map
    """

    def __init__(self):
        super(ResidualModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.res_blocks = nn.ModuleList([ResidualBlock(16) for _ in range(17)])
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 3, kernel_size = 3, stride = 1, padding = 1)
    
    def forward(self, x_input):
        """
        Forward pass 

        Input: x_input - input tensor (1, 3, 16, 16)
        Output: x_output - output tensor (1, 3, 16, 16)
        """

        x = self.conv1(x_input)
        x = self.relu(x)

        for i, block in enumerate(self.res_blocks):
            x = block(x)

        x = self.conv2(x)
        x_output = self.relu(x)

        return x_output
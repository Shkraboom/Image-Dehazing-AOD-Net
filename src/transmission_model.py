import torch
import torch.nn as nn
import cv2
import numpy as np
import h5py

class TransmissionModel(nn.Module):
    """
    СNN for predicting the image transmission map
    """

    def __init__(self):
        super(TransmissionModel, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.conv_3x3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=8, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x_input):
        """
        Forward pass

        Input: x_input - input tensor (1, 3, 16, 16)
        Output: x_output - output tensor (1, 1, 1, 1)
        """

        # print("Input shape: ", x_input.shape)
        x = self.conv_1(x_input)
        x = self.relu(x)
        # print("Shape after conv_1 и relu: ", x.shape)

        x1 = x[:, :4, :, :]
        x2 = x[:, 4:8, :, :]
        x3 = x[:, 8:12, :, :]
        x4 = x[:, 12:, :, :]
        # print("Shape after slicing: ", x1.shape, x2.shape, x3.shape, x4.shape)

        x = torch.max(torch.stack([x1, x2, x3, x4], dim=0), dim=0)[0]
        # print("Shape after pulling: ", x.shape)

        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_7x7 = self.conv_7x7(x)
        # print("Shapes after convolution 3x3, 5x5, 7x7: ", x_3x3.shape, x_5x5.shape, x_7x7.shape)

        x = torch.cat((x_3x3, x_5x5, x_7x7), dim=1)
        # print("Shape after concatenating: ", x.shape)

        x = nn.MaxPool2d(kernel_size=7, stride=1)(x)
        # print("Shape after pulling: ", x.shape)

        x = self.conv2(x)
        x_output = self.relu(x)
        # print("Shape after conv2 и relu: ", x_output.shape)

        return x_output

    @staticmethod
    def guided_filter(im, p, r, eps):
        """
        Guided filter

        Input: im - input snapshot; p - control snapshot; r - window radius; eps - offset for no division by 0
        Output: q - output snapshot with controlled filtering
        """

        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))

        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a * im + mean_b

        return q

    @staticmethod
    def transmission_refine(im, et):
        """
        Improving the quality of the transmission map with a guided filter

        Input: im - input snapshot; et - transmission map
        Output: t - improved transmission map
        """

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray) / 255
        r = 60
        eps = 0.0001
        t = TransmissionModel.guided_filter(gray, et, r, eps)

        return t
################################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Created: 2021-11-27
################################################################################

import torch
import torch.nn as nn
import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """Encoder with a CNN network

        Inputs:
            num_input_channels - Number of input channels of the image. For
                                 FashionMNIST, this parameter is 1
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        # For an intial architecture, you can use the encoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        super().__init__()
        c_hid = num_filters

        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, 4, 1, 2),  # B,  32, 28, 28
            nn.GELU(),
            nn.Conv2d(c_hid,  c_hid, 4, 2, 1),  # B,  32, 14, 14
            nn.GELU(),
            nn.Conv2d(c_hid,  c_hid, 4, 2, 1),  # B,  64,  7, 7
            nn.GELU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(7*7 * c_hid, z_dim)
        self.fc2 = nn.Linear(7*7 * c_hid, z_dim)

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] of type long with values between 0 and 15.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """
        x = x.float() / 15 * 2.0 - 1.0  # Move images between -1 and 1
        x = self.net(x)
        mean = self.fc1(x)
        log_std = self.fc2(x)
        return mean, log_std


class CNNDecoder(nn.Module):
    def __init__(self, num_input_channels: int = 16, num_filters: int = 32,
                 z_dim: int = 20):
        """Decoder with a CNN network.

        Inputs:
            num_input_channels- Number of channels of the image to
                                reconstruct. For a 4-bit FashionMNIST, this parameter is 16
            num_filters - Number of filters we use in the last convolutional
                          layers. Early layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        # For an intial architecture, you can use the decoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        # raise NotImplementedError
        super().__init__()
        c_hid = num_filters

        self.linear = nn.Sequential(
            nn.Linear(z_dim, 7*7 * c_hid),
            nn.GELU()
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(c_hid, c_hid, 4, 2, 1),  # B,  64,  14,  14
            nn.GELU(),
            nn.ConvTranspose2d(c_hid, c_hid, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.GELU(),
            nn.ConvTranspose2d(c_hid, num_input_channels, 4, 1, 2)  # B, 1, 28, 28
        )

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a sigmoid applied on it.
                Shape: [B,num_input_channels,28,28]
        """
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 7, 7)
        x = self.net(x)

        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device


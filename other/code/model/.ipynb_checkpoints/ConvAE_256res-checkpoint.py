import os
import pandas as pd
import pickle
import torch
import numpy as np
import logging
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def out_shape_layer(x_shape, layer):
    """
      Compute output shape  
      Args:
        x: tuple
          Input size (depth, length)
          layer:  The Conv1D layer (nn.Conv1d)
      Returns:
        Tuple of out-channel/out-length
    """
    assert isinstance(layer, nn.Conv1d)
    
    p = layer.padding #if isinstance(layer.padding, tuple) else (layer.padding,)
    k = layer.kernel_size #if isinstance(layer.kernel_size, tuple) else (layer.kernel_size,)
    d = layer.dilation #if isinstance(layer.dilation, tuple) else (layer.dilation,)
    s = layer.stride #if isinstance(layer.stride, tuple) else (layer.stride,)
    
    in_c, in_l = x_shape
    out_c = layer.out_channels
    out_l = 1 + (in_l + 2 * p[0] - (k[0] - 1) * d[0] - 1) // s[0]
    return (out_c, out_l)

class ConvAutoEncoder(nn.Module):
    """
    A 1D Convolutional AutoEncoder
    """
    def __init__(self, x_dim, latent_dim, n_kernels=32, kernel_size=4):
        """
        Initialize parameters of ConvAutoEncoder
        Args:
          x_dim: tuple
            Input dimensions (channels (depth), length)
          latent_dim: int
            latent dimension, bottleneck dimension
          n_kernels: int
            Number of filters (number of output channels)
          kernel_size: int
            Kernel size
        """
        super().__init__()
        channels, length = x_dim
    
        self.channel = channels
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        
        # Encoder
        self.enc_conv_1 = nn.Conv1d(self.channel, self.n_kernels, kernel_size=self.kernel_size , stride=2, padding=1)
        conv_1_shape = out_shape_layer(x_dim, self.enc_conv_1)

        self.enc_conv_2 = nn.Conv1d(self.n_kernels, 2*self.n_kernels, kernel_size=self.kernel_size, stride=2, padding=1)
        conv_2_shape = out_shape_layer(conv_1_shape, self.enc_conv_2)
        
        self.enc_conv_3 = nn.Conv1d(2*self.n_kernels, 4*self.n_kernels, kernel_size=self.kernel_size, stride=2, padding=1)
        conv_3_shape = out_shape_layer(conv_2_shape, self.enc_conv_3)
        
        self.enc_conv_4 = nn.Conv1d(4*self.n_kernels, 8*self.n_kernels, kernel_size=self.kernel_size, stride=2, padding=1)
        conv_4_shape = out_shape_layer(conv_3_shape, self.enc_conv_4)
        
        self.enc_conv_5 = nn.Conv1d(8*self.n_kernels, 16*self.n_kernels, kernel_size=self.kernel_size, stride=2, padding=1)
        conv_5_shape = out_shape_layer(conv_4_shape, self.enc_conv_5)

        self.enc_flatten = nn.Flatten()
        flat = conv_5_shape[0] * conv_5_shape[1]
        
        self.enc_lin = nn.Linear(flat, latent_dim)
        print(flat)
        
        self.dec_lin = nn.Linear(latent_dim, flat)
        
        self.dec_unflatten = nn.Unflatten(dim=-1, unflattened_size=conv_5_shape)
        
        # Decoder
        self.dec_deconv_1 = nn.ConvTranspose1d(16*self.n_kernels, 8*self.n_kernels, kernel_size=self.kernel_size, stride=2, padding=1)

        self.dec_deconv_2 = nn.ConvTranspose1d(8*self.n_kernels, 4*self.n_kernels, kernel_size=self.kernel_size, stride=2, padding=1)
        
        self.dec_deconv_3 = nn.ConvTranspose1d(4*self.n_kernels, 2*self.n_kernels, kernel_size=self.kernel_size, stride=2, padding=1)
        
        self.dec_deconv_4 = nn.ConvTranspose1d(2*self.n_kernels, self.n_kernels, kernel_size=self.kernel_size, stride=2, padding=1)
        
        self.dec_deconv_5 = nn.ConvTranspose1d(self.n_kernels, self.channel, kernel_size=self.kernel_size, stride=2, padding=1)


    def encode(self, x):
        """
        Encoder
        Args:
          x: torch.tensor (input features)
        Returns:
          h: torch.tensor (encoded output)
        """
        l1 = F.relu(self.enc_conv_1(x)) 
        l2 = F.relu(self.enc_conv_2(l1)) 
        l3 = F.relu(self.enc_conv_3(l2)) 
        l4 = F.relu(self.enc_conv_4(l3)) 
        l5 = self.enc_conv_5(l4)
        fl = self.enc_flatten(l5)
        lin1 = self.enc_lin(fl)
        return lin1

    def decode(self, h):
        """
        Decoder
        Args:
          h: torch.tensor (encoded output)
        Returns:
          x_prime: torch.tensor (decoded output)
        """
        lin2 = self.dec_lin(h)
        ufl = self.dec_unflatten(lin2)
        h1 = F.relu(self.dec_deconv_1(ufl))
        h2 = F.relu(self.dec_deconv_2(h1))
        h3 = F.relu(self.dec_deconv_3(h2))
        h4 = F.relu(self.dec_deconv_4(h3))
        x_prime = self.dec_deconv_5(h4)
        
        return x_prime

    def forward(self, x):
        """
        Forward pass
        Args:
          x: torch.tensor (input features)
        Returns: (decoded output)
        """
        return self.decode(self.encode(x))

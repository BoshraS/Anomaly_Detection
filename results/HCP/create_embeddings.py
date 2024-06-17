import sys
sys.path.insert(1, '../../code/')
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import tractogramReader as tr
import time
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import random
import nibabel as nib
from model import ConvAE_256res
from utils import *
from clustering import Kmeans

def main():
    
    model_path = sys.argv[1]
    inp_tractogram = sys.argv[2]
    encoded_inp_tractogram = sys.argv[3]
    
    res = 256
    data_shape = [3, res]
    latent_dim = 64
    n_kernels = 32
    kernel_size = 4
    learning_rate = 1e-3
    batch_size = 256
    device = set_device()
    RANDOM_SEED = 42

    # loading model
    model = ConvAE_256res.ConvAutoEncoder(x_dim=data_shape, latent_dim=latent_dim, n_kernels=n_kernels, kernel_size=kernel_size)
    model.load_state_dict(torch.load(model_path)) 
    model.to(device)
    model.eval()
    
    # reading and encoding streamlines
    encoded_streamlines = read_encode_tractogram(model, inp_tractogram, encoded_inp_tractogram, batch_size, device)
    
if __name__== "__main__":
    main()
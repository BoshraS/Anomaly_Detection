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
    cluster_folder = sys.argv[2]
    inp_tractogram = sys.argv[3]
    encoded_inp_tractogram = sys.argv[4]
    thr = sys.argv[5]
    
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
    
    # clustering
    LS = torch.reshape(encoded_streamlines, (encoded_streamlines.shape[0], encoded_streamlines.shape[1])).cpu()
    LS_keys = list(np.arange(LS.shape[0]))
    LS_dict = {LS_keys[i]: LS[i] for i in tqdm(range(len(LS_keys)))}

    LS_dict_update = LS_dict
    clusters = {}
    label = 0
    
    while tqdm(len(LS_dict_update)>1):
        
        # get the existing keys of the updated list
        
        keys = np.array(list(LS_dict_update.keys()))
        row_to_compare = LS[keys[0],:]
        dist = torch.norm(LS[keys,:]-row_to_compare, dim=1)
        
        # filtering the keys based on the distance threshold
        
        filtered_keys = keys[dist < int(thr)]
        LS_dict_update = {key: value for key, value in LS_dict_update.items() if key not in filtered_keys}
        clusters[label] = filtered_keys
        label += 1
        clusters[label] = keys

    if (len(LS_dict_update)==1):
        clusters[label] = np.array(list(LS_dict_update.keys()))

    clusters_out_file = cluster_folder + '/cluster_centers_thr'+thr
    cluster_centers = np.zeros((len(clusters),64), np.float32)
    
    for i in range(len(clusters)):
        cluster_centers[i] = LS[list(clusters[i])].mean(axis=0)

    cluster_centers.astype(np.float32).tofile(clusters_out_file)
    
if __name__== "__main__":
    main()
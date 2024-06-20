import os
import torch
import sys
sys.path.insert(1, '../../code/')
import numpy as np
from tqdm import tqdm
from datetime import datetime
import tractogramReader as tr
from model import ConvAE_256res
from utils import *
from joblib import cpu_count
import time
import random
import sys

def main():
    
    model_path = sys.argv[1]
    cluster_centers = sys.argv[2]
    inp_tractogram = sys.argv[3]
    out_distances = sys.argv[4]

    device = set_device()
    RANDOM_SEED = 42
    res = 256
    data_shape = [3, res]
    latent_dim = 64
    n_kernels = 32
    kernel_size = 4
    learning_rate = 1e-3
    batch_size = 256

    latent_space_loader = np.fromfile(cluster_centers, dtype=np.float32)
    latent_space_loader = latent_space_loader.reshape(-1,64)
    latent_space_loader = torch.from_numpy(latent_space_loader).to(device)

    model = ConvAE_256res.ConvAutoEncoder(x_dim=data_shape, latent_dim=latent_dim, n_kernels=n_kernels, kernel_size=kernel_size)
    model.load_state_dict(torch.load(model_path)) #map_location=torch.device('cpu')
    model.to(device)
    model.eval()
    
    def min_dist_cal_all(trk_all_lat):

        min_dist = torch.tensor([],dtype=torch.float32, device=device)
        min_dist = min_dist.new_full((trk_all_lat.shape[0],1), float('inf'), device=device) #, dtype=torch.float32

        for n in tqdm(range(trk_all_lat.size(0))):
            tmp  = trk_all_lat[n:n+1,:].repeat(latent_space_loader.shape[0],1)-latent_space_loader
            min_dist[n] = torch.sqrt(torch.sum(torch.pow(tmp,2),axis=1)).min()

        return min_dist.to("cpu")

    tractogram  = tr.T(inp_tractogram.encode("utf-8"))
    N = tractogram.cntStreamline()

    trk  = np.empty([N,3,res], dtype=np.float32)
    for n in tqdm(range(N)):
        trk[n,:,:] = np.asarray(tractogram.read(n)).transpose().reshape([1, 3, res]).astype(np.float32)

    trk  = torch.from_numpy(trk).to(device)
    trk_enc  = encode_strms(model, trk, batch_size, device=device, random_seed=RANDOM_SEED)

    with torch.no_grad():
        min_dist = min_dist_cal_all(trk_enc)
        min_dist.numpy().tofile(out_distances)

    
if __name__== "__main__":
    main();
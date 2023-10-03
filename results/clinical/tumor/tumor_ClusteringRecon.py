import sys
sys.path.insert(1, '../../../code/')
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
    
    device = set_device()
    RANDOM_SEED = 42
    res = 256
    data_shape = [3, res]
    latent_dim = 64
    n_kernels = 32
    kernel_size = 4
    batch_size = 256
    kmeans_batchSize= 1024
    
    model_path = '../../../code/saved_model/AE45.pth'

    model = ConvAE_256res.ConvAutoEncoder(x_dim=data_shape, latent_dim=latent_dim, n_kernels=n_kernels, kernel_size=kernel_size)
    model.load_state_dict(torch.load(model_path)) 
    model.to(device)
    model.eval()

    
    # FiberCup kmeans clustering
    n_clusters = [1000, 10000, 100000]
    start = time.time()
    
    LS = torch.load('HCP_Embeddings/HCP_38sub_N256_L1_64ld_latentSpace.pt', map_location=torch.device('cpu'))
    LS = torch.reshape(LS,(LS.shape[0], LS.shape[1])).numpy()
    
    for _, c in enumerate(n_clusters):
        kmeans_model = Kmeans.MiniBatchKMeansCluster(n_clusters=c, batch_size=kmeans_batchSize, random_state=0)
        kmeans_model.fit(LS)
        centers = kmeans_model.get_cluster_centers()
        clusters_out_file = 'HCP_Clusters/centers_'+str(c)+'_38sub_L1_N256_synTrain_lt'
        centers.astype(np.float32).tofile(clusters_out_file)
    print("Runtime: %s seconds" % (time.time() - start))
    
    for _, c in enumerate(n_clusters):
        output_recTCK_path = 'HCP_Clusters/centers_'+str(c)+'_38sub_L1_N256_synTrain_lt_RecTCK.tck'
        cluster_centers_path = 'HCP_Clusters/centers_'+str(c)+'_38sub_L1_N256_synTrain_lt'
        cluster_centers = np.fromfile(cluster_centers_path, dtype=np.float32)
        cluster_centers = cluster_centers.reshape(-1,64)
        #cluster_centers = torch.from_numpy(cluster_centers).to(device)
        reconstruct_clusterCenters(model=model, 
                                   output_recTCK=output_recTCK_path, 
                                   cluster_centers=cluster_centers, 
                                   res=256, 
                                   device='cpu')
        
if __name__== "__main__":
    main()
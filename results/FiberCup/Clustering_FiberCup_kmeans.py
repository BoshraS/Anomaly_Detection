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
    dataset_path = sys.argv[2]
    out_folder = sys.argv[3]
    
    device = set_device()
    RANDOM_SEED = 42
    res = 256
    data_shape = [3, res]
    latent_dim = 64
    n_kernels = 32
    kernel_size = 4
    batch_size = 256
    kmeans_batchSize= 1024

    tractogram = tr.T(dataset_path.encode("utf-8"))
    model = ConvAE_256res.ConvAutoEncoder(x_dim=data_shape, latent_dim=latent_dim, n_kernels=n_kernels, kernel_size=kernel_size)
    model.load_state_dict(torch.load(model_path)) 
    model.to(device)
    model.eval()
    # Reading the streamlines
    streamlines =  np.empty([tractogram.cntStreamline(),3,res])
    for n in range(tractogram.cntStreamline()):
        streamlines[n] = np.asarray(tractogram.read(n)).transpose()
    streamlines = torch.from_numpy(streamlines).type(torch.FloatTensor)
    # Creating the embeddings
    LS = encode_strms(model, streamlines, batch_size, device=device, random_seed=RANDOM_SEED)
    
    # FiberCup kmeans clustering
    n_clusters = [3, 5, 7, 10, 100]
    start = time.time()
    for _, c in enumerate(n_clusters):
        kmeans_model = Kmeans.MiniBatchKMeansCluster(n_clusters=c, batch_size=kmeans_batchSize, random_state=0)
        kmeans_model.fit(LS.cpu())
        centers = kmeans_model.get_cluster_centers()
        clusters_out_file = out_folder + '/FiberCup_GT_'+str(c)+'clusters'
        output_recTCK_path = out_folder + '/FiberCup_GT_'+str(c)+'clusters_rectTCK.tck'
        centers = centers.astype(np.float32)
        centers.tofile(clusters_out_file)

        reconstruct_clusterCenters(model=model, 
                                   output_recTCK=output_recTCK_path, 
                                   cluster_centers=centers, 
                                   res=256, 
                                   device='cpu')
        
    print("Runtime: %s seconds" % (time.time() - start))
        
if __name__== "__main__":
    main();
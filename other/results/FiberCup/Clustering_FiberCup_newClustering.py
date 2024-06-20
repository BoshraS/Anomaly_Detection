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
    streamlines = torch.from_numpy(streamlines).type(torch.FloatTensor)#.to(device)
    # Creating the embeddings
    LS = encode_strms(model, streamlines, batch_size, device=device, random_seed=RANDOM_SEED).cpu()
    
    LS_keys = list(np.arange(LS.shape[0]))
    LS_dict = {LS_keys[i]: LS[i] for i in tqdm(range(len(LS_keys)))}
    
    thrs = [50, 85, 200, 900]
    start = time.time()
    
    for _, thr in enumerate(thrs):
        LS_dict_update = LS_dict
        clusters = {}
        label = 0
        
        while tqdm(len(LS_dict_update)>1):
            # get the existing keys of the updated list
            keys = np.array(list(LS_dict_update.keys()))
            row_to_compare = LS[keys[0],:]
            dist = torch.norm(LS[keys,:]-row_to_compare, dim=1)
            # filtering the keys based on the distance threshold
            filtered_keys = keys[dist < thr]
            clusters[label] = filtered_keys
            label += 1
            #clusters[label] = keys
            LS_dict_update = {key: value for key, value in LS_dict_update.items() if key not in filtered_keys}

        if (len(LS_dict_update)==1):
            clusters[label] = np.array(list(LS_dict_update.keys()))
    
        clusters_out_file = out_folder + '/newClustering_FiberCup_centers_thr'+str(thr)
        output_recTCK_path = out_folder + '/FiberCup_GT_new_thr'+str(thr)+'_clusters_rectTCK.tck'

        cluster_centers = np.zeros((len(clusters),64), np.float32)
        for i in range(len(clusters)):
            cluster_centers[i] = LS[list(clusters[i])[0]]#.mean(axis=0)

        cluster_centers = cluster_centers.astype(np.float32)
        cluster_centers.tofile(clusters_out_file)
        
        reconstruct_clusterCenters(model=model, 
                                   output_recTCK=output_recTCK_path, 
                                   cluster_centers=cluster_centers, 
                                   res=256, 
                                   device='cpu')
    
    print("Runtime: %s seconds" % (time.time() - start))
        
if __name__== "__main__":
    main();
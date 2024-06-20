import sys
sys.path.insert(1, '../../code/')
import numpy as np
import torch
import os
import glob
from utils import *

def main():
    source_dir = sys.argv[1]
    cluster_dir = sys.argv[2]
    thr = sys.argv[3]
    
    device = set_device()

    def load_tensors(file_path):
        return torch.load(file_path)

    LS = []

    file_paths = glob.glob(os.path.join(source_dir, "*_trekker_LS_10M_256.pt"))

    for file_path in file_paths:
        print(f"Loading data from {file_path}")
        subject_tensors = load_tensors(file_path)
        # Assuming subject_tensors is a PyTorch tensor, convert it to numpy array
        subject_tensors_np = subject_tensors
        LS.append(subject_tensors_np)

    # Combine all tensors
    LS = torch.cat(LS, dim=0)#.to(device)

    #LS = torch.reshape(encoded_streamlines, (encoded_streamlines.shape[0], encoded_streamlines.shape[1])).cpu()
    LS_keys = list(np.arange(LS.shape[0]))
    LS_dict = {LS_keys[i]: LS[i] for i in tqdm(range(len(LS_keys)))}

    LS_dict_update = LS_dict
    clusters = {}
    label = 0

    while tqdm(len(LS_dict_update)>1):

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

    clusters_out_file = cluster_dir + '/cluster_centers_thr'+thr
    cluster_centers = np.zeros((len(clusters),64), np.float32)

    for i in range(len(clusters)):
        cluster_centers[i] = LS[list(clusters[i])].mean(axis=0)

    cluster_centers.astype(np.float32).tofile(clusters_out_file)
    
if __name__== "__main__":
    main()

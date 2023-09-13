import numpy as np
from tqdm import tqdm
from datetime import datetime
from utils import *
import time
import torch
from joblib import cpu_count
from sklearn.cluster import MiniBatchKMeans
def main():
    latent_space_loader = torch.load('HCP_38sub_N256_MSEL_64ld_latentSpace.pt', map_location=torch.device('cpu'))
    latent_space_loader = torch.reshape(latent_space_loader,(latent_space_loader.shape[0], latent_space_loader.shape[1])).numpy()

    start = time.time()
    print(cpu_count())
    km = MiniBatchKMeans(n_clusters=10000, random_state=0, batch_size=25600, verbose=True).fit(latent_space_loader)
    centers = km.cluster_centers_
    centers.tofile("centers_100K_38sub_N256_synTrain_lt")

    print("Runtime: %s seconds" % (time.time() - start))

if __name__== "__main__":
      main();

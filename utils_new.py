import os
import torch
import numpy as np
import random
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from joblib import cpu_count
from sklearn.cluster import KMeans
import time
from sklearn.cluster import MiniBatchKMeans


def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled. \n")
    else:
        print("GPU is enabled. \n")
    return device

def min_dist(sample, trainData_to_Lspace):
    return torch.nn.functional.pairwise_distance(sample,trainData_to_Lspace).min()
    

def seed_all(seed: int = 42):
    
    """Seed all random number generators."""
    print("Using Seed Number {}".format(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def seed_worker(_worker_id):
    
    """Seed a worker with the given ID."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def visualise_tck(tck_org, tck_recon):
    
    # look also https://plotly.com/python/streamtube-plot/
    x1 = tck_org[0,:]
    y1 = tck_org[1,:]
    z1 = tck_org[2,:]
    
    x2 = tck_recon[0,:]
    y2 = tck_recon[1,:]
    z2 = tck_recon[2,:]

    plt.rcParams["figure.figsize"] = [60, 60]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x1, y1, z1, s=400, c=z1)
    ax.plot(x1, y1, z1, color='black', linewidth=20)
    ax.scatter(x2, y2, z2, s=400, c=z2)
    ax.plot(x2, y2, z2, color='red', linewidth=20)
    plt.show()
    
    return 0

def streamline_len(strm):
    # strm with shape [3,res]
    lengths = np.zeros(strm.shape[1],)
    for i in range(0, strm.shape[1]-1):
        x1 = strm[0, i]
        x2 = strm[0, i+1]
        y1 = strm[1, i]
        y2 = strm[1, i+1]
        z1 = strm[2, i]
        z2 = strm[2, i+1]
        lengths[i] = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)

    return np.sum(lengths)

def plot_ranked_comparison(measure1: np.ndarray, measure2: np.ndarray, label1: str = 'x', label2: str = 'y'):
    g = sns.jointplot(x=measure1,y=measure2,kind="reg")
    r, p = stats.spearmanr(measure1,measure2)
    g.ax_joint.annotate(f'$r_s = {r:.3f}, p = {p:e}$',
                        xy=(0.1, 0.9), xycoords='axes fraction',
                        ha='left', va='center',
                        bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'}
                       )
    g.ax_joint.scatter(measure1, measure2)
    g.set_axis_labels(xlabel=label1, ylabel=label2, size=15)
    return g

    
def encode_strms(model, streamlines, batch_size, device, RANDOM_SEED=42):
    seed_all(RANDOM_SEED)
    with torch.no_grad():
        temp = []
        seed_all(seed=RANDOM_SEED)
        inp_batched = torch.split(streamlines, batch_size)
        for batch_id, strm_coor_batch in enumerate(inp_batched):
            strm_coor_batch = strm_coor_batch.to(device)
            strm_coor_batch_encode = model.encode(strm_coor_batch)
            if batch_id==0:
                temp = strm_coor_batch_encode.detach()
                continue
            temp = torch.cat((temp,strm_coor_batch_encode.detach()))
    return temp


def MDF_strms_dist(strm1, strm2):
    
    #Computing the direct distance
    direct_dists = torch.zeros(strm1.shape[1],)
    for i in range(strm1.shape[1]):
        x1 = strm1[0, i]
        x2 = strm2[0, i]
        y1 = strm1[1, i]
        y2 = strm2[1, i]
        z1 = strm1[2, i]
        z2 = strm2[2, i]
        direct_dists[i] = torch.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    direct_dist = direct_dists.mean()
    
    # computing the flipped distance
    flipped_dists = torch.zeros(strm1.shape[1],)
    strm2 = torch.flip(strm2, [1])
    for i in range(strm1.shape[1]):
        x1 = strm1[0, i]
        x2 = strm2[0, i]
        y1 = strm1[1, i]
        y2 = strm2[1, i]
        z1 = strm1[2, i]
        z2 = strm2[2, i]
        flipped_dists[i] = torch.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    flipped_dist = flipped_dists.mean()
    
    return min(direct_dist,flipped_dist)

def kmeans_clustering(LS, num_clusters, out_file_name):
    start = time.time()
    km = KMeans(n_clusters=num_clusters, random_state=0, verbose=True, max_iter=1000).fit(LS.cpu())
    centers = km.cluster_centers_
    np.save(out_file_name,centers)
    print("Runtime: %s seconds" % (time.time() - start))
    

def miniBatch_kmeans_clustering(LS, num_clusters, out_file_name, batch_size=1024):
    
    start = time.time()
    km = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=batch_size, verbose=True).fit(LS.cpu())
    centers = km.cluster_centers_
    np.save(out_file_name,centers)
    print("Runtime: %s seconds" % (time.time() - start))

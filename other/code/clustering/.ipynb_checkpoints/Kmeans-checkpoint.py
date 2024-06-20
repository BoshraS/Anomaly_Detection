import numpy as np
from tqdm import tqdm
from datetime import datetime
from utils import *
import time
import torch
from joblib import cpu_count
from sklearn.cluster import MiniBatchKMeans

class MiniBatchKMeansCluster:
    def __init__(self, n_clusters=8, batch_size=100, max_iter=100, random_state=0):
        """
        Initialize the MiniBatchKMeansCluster instance.
        """
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(self, data):
        """
        Fit the Mini-Batch K-Means model to the given data.
        """
        self.model = MiniBatchKMeans(n_clusters=self.n_clusters, 
                                     batch_size=self.batch_size, 
                                     max_iter=self.max_iter, 
                                     random_state=self.random_state
                                     )
        self.model.fit(data)

    def predict(self, data):
        """
        Predict cluster assignments for new data points.
        """
        if self.model is None:
            raise ValueError("The model has not been trained. Call the 'fit' method first.")
        return self.model.predict(data)

    def get_cluster_centers(self):
        """
        Get the coordinates of the cluster centers.
        """
        if self.model is None:
            raise ValueError("The model has not been trained. Call the 'fit' method first.")
        return self.model.cluster_centers_

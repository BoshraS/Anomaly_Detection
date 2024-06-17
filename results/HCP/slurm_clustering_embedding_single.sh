#!/bin/bash -eux
# =====================================================================================================================
# slurm_clustering_embeddings.sh - A script to cluster a whole-brain tractogram, using our 
# trained model to first encode the tractogram 
#
# Usage:
#      sbatch slurm_clustering_embeddings.sh
# Description:
# The script will perform clustering on a encoded tractogram using the specified threshold ("thr") and save the 
# results in "cluster_folder". To achieve this, the model will be loaded in the main script from the "model_path" input. 
# The whole brain tractogram ("inp_tractogram") is then read and encoded using the "read_encode_tractogram" function from 
# Anomaly-Detection/code/utils.py. The encoded tractogram is then saved as "encoded_inp_tractogram".
#
# =====================================================================================================================

#SBATCH --job-name=tck_ae_anomaly
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boshra.shams@guest.hpi.de
#SBATCH --partition=gpua100 
#SBATCH --cpus-per-task=20 
#SBATCH --gpus=a100 
#SBATCH --time=3-00:00:00

model_path=../../code/saved_model/AE45.pth 
cluster_folder=HCP_clusters
inp_tractogram=HCP_trekker/100307_trekker_10M_256.tck
encoded_inp_tractogram=HCP_trekker_LS/100307_trekker_LS_10M_256.pt
thr=10000
python clustering_embeddings.py ${model_path} ${cluster_folder} ${inp_tractogram} ${encoded_inp_tractogram} ${thr} 
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

# Set the right python environment which has pytorch and TractogramReader packages installed
source /home/baran/Work/code/ML/env/bin/activate ML

# python saveJitScript.py



streamlines=/home/baran/Work/project/Boshra/code/Anomaly_Detection/results/FiberCup/FiberCup_GT/groundTruth_MNI_N256.tck
streamlines_features=test/FC_clusters/FC_features.pt
out_centers=test/FC_clusters/FC_centers

streamlines=/home/baran/Work/code/anomalyMapper/test/one/ten.tck
streamlines_features=/home/baran/Work/code/anomalyMapper/test/one/ten_python.pt
out_centers=/home/baran/Work/code/anomalyMapper/test/one/ten_center

: '
streamlines=/home/baran/Work/code/anomalyMapper/test/one/one.tck
streamlines_features=/home/baran/Work/code/anomalyMapper/test/one/one_python.pt
out_centers=/home/baran/Work/code/anomalyMapper/test/one/one_center
'

python cluster.py \
-i ${streamlines} \
-f ${streamlines_features} \
-o ${out_centers}.tck \
-c ${out_centers}.pt \
-t 1 \
-m models/AE45.pth



: '
streamlines=/home/baran/Work/project/Boshra/code/Anomaly_Detection/results/FiberCup/FiberCup_GT/groundTruth_MNI_N256.tck
streamlines_features=test/FC_clusters/FC_features.pt
out_centers=test/FC_clusters/FC_centers


python cluster.py \
-i ${streamlines_features} \
-o ${out_centers}.tck \
-c ${out_centers}.pt \
-t 40 \
-m models/AE45.pth

nipt tractogram convert -f \
${out_centers}.tck \
${out_centers}.vtk
'

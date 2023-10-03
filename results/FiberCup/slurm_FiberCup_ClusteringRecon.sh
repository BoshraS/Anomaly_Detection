#!/bin/bash -eux

#SBATCH --job-name=tck_ae_anomaly

#SBATCH --mail-type=ALL

#SBATCH --mail-user=boshra.shams@guest.hpi.de

#SBATCH --partition=gpupro # -p

#SBATCH --cpus-per-task=10 # -c

#SBATCH --gpus=a40 # -p

#SBATCH --time=3-00:00:00

#SBATCH --output=/dhc/home/boshra.shams/FiberCup_clustering.log 

python FiberCup_ClusteringRecon.py
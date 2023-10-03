#!/bin/bash -eux

#SBATCH --job-name=tck_ae_anomaly

#SBATCH --mail-type=ALL

#SBATCH --mail-user=boshra.shams@guest.hpi.de

#SBATCH --partition=gpupro# -p

#SBATCH --cpus-per-task=4 # -c

#SBATCH --gpus=a40 # -p

#SBATCH --time=3-00:00:00

#SBATCH --output=/dhc/home/boshra.shams/ISMRM_GT_anomaly.log 

model_path=../../code/saved_model/AE45.pth
MNI_template=../../code/MNI152_T1_1mm_brain.nii.gz
inp_tractogram=ISMRM_GT_HCP/wb_MNI_N256.tck

for i in {1000,10000,100000} ; do 
    echo "Processing file number ${i}"

    out_distances=ISMRM_GT_Clusters/dist_ISMRM_GT_${i}clusters
    cluster_centers=ISMRM_GT_Clusters/ISMRM_GT_${i}clusters
    anomaly_MNI=ISMRM_GT_Clusters/ISMRM_GT_${i}clusters_anomaly_MNI.nii.gz
    inp_MNI=ISMRM_GT_Clusters/ISMRM_GT_${i}clusters_MNI.nii.gz
    anomaly_ratio_MNI=ISMRM_GT_Clusters/ISMRM_GT_${i}clusters_anomaly_ratio_MNI.nii.gz

    python getDistance.py ${model_path} ${cluster_centers} ${inp_tractogram} ${out_distances}

    ../../code/nipt tractogram map2image \
    $inp_tractogram \
    ${anomaly_MNI} \
    --feature segmentLength \
    --weights ${out_distances} \
    --template ${MNI_template} \
    -f
    ../../code/nipt tractogram map2image \
    ${inp_tractogram} \
    ${inp_MNI} \
    --feature segmentLength \
    --template ${MNI_template} \
    -f
    ../../code/nipt image math -f ${anomaly_ratio_MNI} ${anomaly_MNI} div ${inp_MNI}

done 
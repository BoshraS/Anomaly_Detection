#!/bin/bash -eux

#SBATCH --job-name=tck_ae_anomaly

#SBATCH --mail-type=ALL

#SBATCH --mail-user=boshra.shams@guest.hpi.de

#SBATCH --partition=gpupro # -p

#SBATCH --cpus-per-task=4 # -c

#SBATCH --gpus=a40 # -p

#SBATCH --time=3-00:00:00

#SBATCH --output=/dhc/home/boshra.shams/anomaly_FiberCup_GT.log 

model_path=../../code/saved_model/AE45.pth
MNI_template=../../code/MNI152_T1_1mm_brain.nii.gz
inp_tractogram=FiberCup_GT/groundTruth_MNI_N256.tck

for i in {3,5,7,10,100} ; do 
    echo "Processing file number ${i}"
    
    out_distances=FiberCup_Clustering/dist_FiberCup_GT_${i}clusters
    cluster_centers=FiberCup_Clustering/FiberCup_GT_${i}clusters
    anomaly_MNI=FiberCup_Clustering/FiberCup_GT_${i}clusters_anomaly_MNI.nii.gz
    inp_MNI=FiberCup_Clustering/FiberCup_GT_${i}clusters_MNI.nii.gz
    anomaly_ratio_MNI=FiberCup_Clustering/FiberCup_GT_${i}clusters_anomaly_ratio_MNI.nii.gz
    
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
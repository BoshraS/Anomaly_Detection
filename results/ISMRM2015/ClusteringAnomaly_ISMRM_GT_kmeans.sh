#!/bin/bash -eux

#SBATCH --job-name=tck_ae_anomaly
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boshra.shams@guest.hpi.de
#SBATCH --partition=gpupro# -p
#SBATCH --cpus-per-task=4 # -c
#SBATCH --gpus=a40 # -p
#SBATCH --time=3-00:00:00

model_path=../../code/saved_model/AE45.pth
MNI_template=../../code/MNI152_T1_1mm_brain.nii.gz
inp_tractogram=ISMRM_GT/wb_MNI_N256.tck
out_folder=ISMRM_GT_Clusters

python ISMRM_Clustering_kmeans.py ${model_path} ${inp_tractogram} ${out_folder}

for i in {1000,3720,10000,100000} ; do 
    echo "Processing file number ${i}"

    out_distances=${out_folder}/dist_ISMRM_GT_${i}clusters
    cluster_centers=${out_folder}/ISMRM_GT_${i}clusters
    anomaly_MNI=${out_folder}/ISMRM_GT_${i}clusters_anomaly_MNI.nii.gz
    inp_MNI=${out_folder}/ISMRM_GT_${i}clusters_MNI.nii.gz
    anomaly_ratio_MNI=${out_folder}/ISMRM_GT_${i}clusters_anomaly_ratio_MNI.nii.gz

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
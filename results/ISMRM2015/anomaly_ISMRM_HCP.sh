#!/bin/bash -eux

#SBATCH --job-name=tck_ae_anomaly

#SBATCH --mail-type=ALL

#SBATCH --mail-user=boshra.shams@guest.hpi.de

#SBATCH --partition=gpupro# -p

#SBATCH --cpus-per-task=10 # -c

#SBATCH --gpus=a40 # -p

#SBATCH --time=3-00:00:00

#SBATCH --output=/dhc/home/boshra.shams/ISMRM_HCP_anomaly.log 

model_path=../../code/saved_model/AE45.pth
MNI_template=../../code/MNI152_T1_1mm_brain.nii.gz

for i in {1000,10000,100000} ; do 
    echo "Processing file number ${i}"
    for param in {0,1,2,3,4,5,6,7,8} ; do
        inp_tractogram=ISMRM_GT_HCP/hcp_wb_params_${param}_MNI_N256.tck
        out_distances=ISMRM_GT_HCP/dist_HCP_wb_params_${param}_${i}clusters
        cluster_centers=ISMRM_GT_Clusters/ISMRM_GT_${i}clusters
        anomaly_MNI=ISMRM_GT_HCP/HCP_wb_params_${param}_${i}clusters_anomaly_MNI.nii.gz
        inp_MNI=ISMRM_GT_HCP/HCP_wb_params_${param}_${i}clusters_MNI.nii.gz
        anomaly_ratio_MNI=ISMRM_GT_HCP/HCP_wb_params_${param}_${i}clusters_anomaly_ratio_MNI.nii.gz

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
done 
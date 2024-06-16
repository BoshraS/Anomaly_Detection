#!/bin/bash -eux

#SBATCH --job-name=tck_ae_anomaly
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boshra.shams@guest.hpi.de
#SBATCH --partition=gpupro# -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --gpus=a40 # -p
#SBATCH --time=3-00:00:00

model_path=../../code/saved_model/AE45.pth
MNI_template=../../code/MNI152_T1_1mm_brain.nii.gz
out_folder=ISMRM_GT_HCP

for param in {0,1,2,3,4,5,6,7,8,9} ; do
    inp_tractogram=${out_folder}/hcp_wb_params_${param}_MNI_N256.tck
    out_distances=${out_folder}/dist_HCP_wb_params_${param}_newClus50
    cluster_centers=ISMRM_GT_Clusters/newClustering_ISMRM_GT_centers_thr50
    anomaly_MNI=${out_folder}/HCP_wb_params_${param}_newClus50_anomaly_MNI.nii.gz
    inp_MNI=${out_folder}/HCP_wb_params_${param}_newClus50_MNI.nii.gz
    anomaly_ratio_MNI=${out_folder}/HCP_wb_params_${param}_newClus50_anomaly_ratio_MNI.nii.gz

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
#!/bin/bash -eux

#SBATCH --job-name=tck_ae_anomaly

#SBATCH --mail-type=ALL

#SBATCH --mail-user=boshra.shams@guest.hpi.de

#SBATCH --partition=gpua100# -p

#SBATCH --cpus-per-task=4 # -c

#SBATCH --gpus=a100 # -p

#SBATCH --time=3-00:00:00

#SBATCH --output=/dhc/home/boshra.shams/Tumor_patient_anomaly.log 

root=../../../code
model_path=${root}/saved_model/AE45.pth
MNI_template=${root}/MNI152_T1_1mm_brain.nii.gz


for c in {1000,10000} ; do
    cluster_centers=HCP_Clusters/centers_${c}_38sub_L1_N256_synTrain_lt
    for i in {S161,S198,S214,S216} ; do 
        echo "Processing file number ${i}"
        inp_tractogram=language_patients/${i}/11M_minlen60_N256_MNI.tck
        out_distances=language_patients/${i}/dist_${i}__${c}clusters
        anomaly_MNI=language_patients/${i}/${i}_${c}clusters_anomaly_MNI.nii.gz
        inp_MNI=language_patients/${i}/${i}_${c}clusters_MNI.nii.gz
        anomaly_ratio_MNI=language_patients/${i}/${i}_${c}clusters_anomaly_ratio_MNI.nii.gz

        python getDistance.py ${model_path} ${cluster_centers} ${inp_tractogram} ${out_distances}

        ${root}/nipt tractogram map2image \
        $inp_tractogram \
        ${anomaly_MNI} \
        --feature segmentLength \
        --weights ${out_distances} \
        --template ${MNI_template} \
        -f
        ${root}/nipt tractogram map2image \
        ${inp_tractogram} \
        ${inp_MNI} \
        --feature segmentLength \
        --template ${MNI_template} \
        -f
        ${root}/nipt image math -f ${anomaly_ratio_MNI} ${anomaly_MNI} div ${inp_MNI}
    done
done 
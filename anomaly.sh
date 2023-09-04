#!/bin/bash -eux

#SBATCH --job-name=tck_ae_anomaly

#SBATCH --mail-type=ALL

#SBATCH --mail-user=boshra.shams@guest.hpi.de

#SBATCH --partition=gpua100 # -p

#SBATCH --cpus-per-task=10 # -c

#SBATCH --gpus=a100 # -p

#SBATCH --time=3-00:00:00

#SBATCH --output=/dhc/home/boshra.shams/anomaly_patient.log 

root=/dhc/home/boshra.shams/CNN_AE/AnomalyMap_HCP_synTrain/AnomalyMapping_patient/
for i in {1k,10k,100k} ; do 
    echo "Anomaly mapping for ${i} HCP data number of clusters"
    for sub_ID in {S161,S198,S216,S214} ; do 
        echo "Processing subject ID: ${sub_ID}"
        inp_tractogram=${root}${sub_ID}/11M_minlen60_N256.tck
        out_distances=${root}${sub_ID}/dist_11M_minlen60_N256_${i}clusters
        model_path=${root}AE45.pth
        cluster_centers=${root}${sub_ID}/centers_${i}_38sub_L1_N256_synTrain_lt
        anomaly_MNI=${root}${sub_ID}/11M_minlen60_N256_${i}clusters_anomaly_MNI.nii.gz
        MNI_template=${root}MNI152_T1_1mm_brain.nii.gz
        inp_MNI=${root}${sub_ID}/11M_minlen60_N256_${i}clusters_MNI.nii.gz
        anomaly_ratio_MNI=${root}${sub_ID}/11M_minlen60_${i}clusters_anomaly_ratio_MNI.nii.gz

        python getDistance.py ${model_path} ${cluster_centers} ${inp_tractogram} ${out_distances}

        ./nipt tractogram map2image \
        $inp_tractogram \
        ${anomaly_MNI} \
        --feature segmentLength \
        --weights ${out_distances} \
        --template ${MNI_template} \
        -f
        ./nipt tractogram map2image \
        ${inp_tractogram} \
        ${inp_MNI} \
        --feature segmentLength \
        --template ${MNI_template} \
        -f
        ./nipt image math -f ${anomaly_ratio_MNI} ${anomaly_MNI} div ${inp_MNI}
    done
done 
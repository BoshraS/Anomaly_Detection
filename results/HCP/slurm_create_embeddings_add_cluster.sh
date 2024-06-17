#!/bin/bash -eux

#SBATCH --job-name=tck_ae_anomaly
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boshra.shams@guest.hpi.de
#SBATCH --partition=gpupro 
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a40
#SBATCH --time=3-00:00:00

model_path=../../code/saved_model/AE45.pth 

SOURCE_DIR="HCP_trekker"
DEST_DIR="HCP_trekker_LS"
CLUSTER_DIR="HCP_clusters"
THR=100

if [ ! -d "$DEST_DIR" ]; then
    echo "Destination directory does not exist. Creating $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

for inp_tractogram in "$SOURCE_DIR"/*.tck; do
    if [ -f "$inp_tractogram" ]; then
        BASENAME=$(basename "$inp_tractogram" .tck)
        encoded_inp_tractogram="${BASENAME}_LS.pt"
        
        encoded_inp_tractogram_path="$DEST_DIR/$encoded_inp_tractogram"
        
        echo "Processing $inp_tractogram"
        echo "Saving as $encoded_inp_tractogram_path"
        
        python create_embeddings.py ${model_path} ${inp_tractogram} ${encoded_inp_tractogram} 

    fi
python read_all_embeddings_clustering.py ${DEST_DIR} ${CLUSTER_DIR} ${THR}

done
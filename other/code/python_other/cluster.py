import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
import ConvAE_256res
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Cluster streamlines using deep learning")

    parser.add_argument('-i', '--input', type=str, help='Input tractogram (.tck) or streamlines features (.pt), i.e. latent space representations')
    parser.add_argument('-o', '--output', type=str, help='Output file to save the cluster centers in .tck format')
    parser.add_argument('-f', '--outputFeatures', type=str, help='Output file for streamline features if the input is a tractogram')
    parser.add_argument('-c', '--outputClusterCenterFeatures', type=str, help='Output file for cluster center features in latent space (.pt)')
    parser.add_argument('-t', '--threshold', type=str, help='Threshold for similar similarity (approx. distance) for clustering')
    parser.add_argument('-d', '--device', type=str, help='GPU or CPU. Default: GPU, if available')
    parser.add_argument('-m', '--model', type=str, help='Streamline encoder-decoder model')

    return parser

def set_device(device):
    if device == "cpu":
        return device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU not found. Using CPU \n")
    return device

def load_model(model_path, device):
    data_shape = [3, 256]
    latent_dim = 64
    n_kernels = 32
    kernel_size = 4

    model = ConvAE_256res.ConvAutoEncoder(x_dim=data_shape, latent_dim=latent_dim, n_kernels=n_kernels, kernel_size=kernel_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    return model

def read_streamlines(inp_tractogram):
    tractogram = nib.streamlines.load(inp_tractogram)
    streamlines = tractogram.streamlines
    streamlines_array = np.array([s.T for s in streamlines])  # Transpose each streamline to shape (3, 256)
    streamlines_tensor = torch.from_numpy(streamlines_array).type(torch.FloatTensor)
    return streamlines_tensor

def encode_strms(streamlines, model, device, batch_size):
    with torch.no_grad():
        temp = []
        inp_batched = torch.split(streamlines, batch_size)
        with tqdm(total=len(inp_batched), desc="Info: Encoding streamlines") as pbar:
            for batch_id, strm_coor_batch in enumerate(inp_batched):
                strm_coor_batch = strm_coor_batch.to(device)
                strm_coor_batch_encode = model.encode(strm_coor_batch)
                if batch_id == 0:
                    temp = strm_coor_batch_encode.detach()
                else:
                    temp = torch.cat((temp, strm_coor_batch_encode.detach()))
                pbar.update(1)
    return temp


def decode_strms(output_file, encoded_strms, model, device):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.to(device)
    model.eval()

    def nextStreamline():
        for i in range(np.size(encoded_strms, 0)):
            latent_strms = encoded_strms[i:i+1, :]
            with torch.no_grad():
                latent = torch.from_numpy(latent_strms).to(device)
                trk = model.decode(latent).cpu().numpy().reshape([3, 256]).astype(np.float32).transpose()
                yield trk

    trk_out = nib.streamlines.tck.TckFile(nib.streamlines.tractogram.LazyTractogram(nextStreamline, None, None, np.eye(4, 4)))
    trk_out.save(output_file)

def encode_tractogram_and_save(model, inp_tractogram, out_encoded_tractogram, batch_size, device):
    print("Info: Reading streamlines", flush=True)
    streamlines = read_streamlines(inp_tractogram)
    streamlines = streamlines.to(device)

    # encode tractogram
    print("Info: Encoding streamlines", flush=True)
    encoded_strms = encode_strms(streamlines, model, device, batch_size)

    # Save the encoded streamlines
    print("Info: Saving encoded streamlines", flush=True)
    torch.save(encoded_strms.cpu(), out_encoded_tractogram)

    return encoded_strms

def main(args):

    # Initialize variables
    LS    = None
    model = None

    # Ensure the output file extension is .tck
    if not args.output.endswith('.tck'):
        print("Error: Output cluster centers must have a .tck extension")
        return

    # Ensure the output file extension is .tck
    if args.outputClusterCenterFeatures:
        if not args.outputClusterCenterFeatures.endswith('.pt'):
            print("Error: Output cluster center features must have a .pt extension")
            return

    # If the input file is a tractogram
    if args.input.endswith('.tck'):
        if args.outputFeatures:
            device = set_device(args.device)  # Set device
            model = load_model(args.model, device)  # Load the model
            LS = encode_tractogram_and_save(model, args.input, args.outputFeatures, 256, device)
            LS = torch.reshape(LS, (LS.shape[0], LS.shape[1])).cpu()
        else:
            print("Error: Output file for features must be specified with '-f' when input is a tractogram.")
            return

    # If the input is latent space representations
    elif args.input.endswith('.pt'):
        print("Info: Reading encoded streamlines", flush=True)
        LS = torch.load(args.input, map_location=torch.device('cpu'))
        device = set_device(args.device)  # Set device
        model = load_model(args.model, device)  # Load the model

    else:
        print("Error: Unknown input type")
        return

    print(LS)

    # Do the clustering
    LS_keys = list(np.arange(LS.shape[0]))
    LS_dict = {LS_keys[i]: LS[i] for i in range(len(LS_keys))}
    clusters = {}
    label = 0

    # Clustering loop
    with tqdm(total=len(LS_dict), desc="Info: Clustering streamlines") as pbar:
        while len(LS_dict) > 1:
            keys = np.array(list(LS_dict.keys()))
            row_to_compare = LS[keys[0], :]
            dist = torch.norm(LS[keys, :] - row_to_compare, dim=1)

            # Filtering the keys based on the distance threshold
            filtered_keys = keys[dist < float(args.threshold)]
            clusters[label] = filtered_keys
            label += 1

            # Update LS_dict by removing the filtered keys
            LS_dict = {key: value for key, value in LS_dict.items() if key not in filtered_keys}

            # Update progress bar
            pbar.update(len(filtered_keys))

    # Handle the case where only one element is left
    if len(LS_dict) == 1:
        clusters[label] = np.array(list(LS_dict.keys()))

    cluster_centers = np.zeros((len(clusters), 64), np.float32)

    for i in range(len(clusters)):
        cluster_centers[i] = LS[list(clusters[i])].mean(axis=0)

    # Save the cluster centers as .tck file
    decode_strms(args.output, cluster_centers, model, device)

    # If the outputClusterCenterFeatures argument is set, save the cluster center features
    if args.outputClusterCenterFeatures:
        cluster_centers.astype(np.float32).tofile(args.outputClusterCenterFeatures)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)

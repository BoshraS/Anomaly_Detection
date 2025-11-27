# Anomap

*Anomap* computes structural anomalies in the white matter by processing tractograms.

## Installation

Step 1. Install the dependencies written below

Step 2. Modify the `build.sh` script

Step 3. Run the `build.sh` script

#### Dependencies
- [Nibrary](https://github.com/nibrary/nibrary)
    - Tested with [v0.4.4](https://github.com/nibrary/nibrary/tree/v0.4.4)
- [LibTorch](https://pytorch.org/)
    - If CUDA version is installed, GPU acceleration will be used.
    - Make sure to install the cxx11 ABI version of LibTorch!
    - Tested with [CUDA v12.9](https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip)

#### Optional
- [Matplotlib](https://matplotlib.org/)
    - Used for testing the streamline encoder-decoder model. If a Python environment with matplotlib available, charts will be plotted, otherwise results are only printed on the terminal.
    


## Usage

`anomap` comes with the following subcommands, which enable the preparation of a reference tractogram that is consider to contain *normal* streamlines. Then given any input tractogram, anomap assings an anomaly score to each streamline by comparing them with the streamlines in the reference tractogram.


| Command    | Description |
|------------|-------------|
| **modelTest**  | compares the pair-wise distance in latent space agains the distance computed using two-sided Hausdorff and minimum average direct-flip (MDF) distances|
| **encode** | computes and saves latent space representation streamlines  |
| **decode** | decodes latent space representations and output a tractogram  |
|**findClusterCenters**| finds streamline cluster centers |
|**score**| assigns an anomaly score to each streamline |
|**toTrack**| maps anomaly scores and/or cluster labels on streamlines |
|**toImg**| maps anomaly scores on an image |




#### Example

```bash
# Path to anomap executable
anomap=build/install/anomap

# Compute latent representations

ref=/data/ref                      # Input folder containing reference tractograms
latent=/data/refLatent             # Output folder for latent representations

${anomap} encode ${ref} ${latent}


# Compute cluster centers
# e.g. with a max distance of ~20 mm between cluster centers

clusterCenters=/data/clusters.clc  # Output path for cluster centers
clusterLabels=/data/clusters.clb   # Output path for cluster labels

${anomap} findClusterCenters ${latent} ${clusterCenters} ${clusterLabels} --maxDist 20


# Compute anomaly scores

input=/data/input.vtk        # Input tractogram for anomaly scoring
output=/data/output          # Output prefix

${anomap} score ${input} ${clusters} ${output}.ano


# Map anomaly scores and cluster labels on tractogram
${anomap} toTrack ${input} -s ${output}.ano -l ${output}.clc ${output}.vtk

# Map anomaly scores on an image
${anomap} toImg ${input} ${output}.ano ${output}.nii.gz

```

## Publications

- Shams B, Vajkoczy P, Picht T, Fekonja L, Aydogan DB. *Deep neural network based visualization
of white matter deformation in glioma patients.* Annual Meeting of the German Society for Neurosurgery,
Berlin, Germany, 2023

- Shams B, Picht T, Vajkoczy P, Fekonja L, Aydogan DB. *Mapping and visualization of white matter
deformation in glioma patients using deep learning.* Organization for Human Brain Mapping Annual
Meeting, Montréal, Canada, 2023

## Anomaly map team

- Shams B
- Fekonja L
- Picht T
- Aydogan DB

#!/bin/bash

echo "Running model test"

anomap=../../build/anomap

inp=random_1K.vtk
# inp=random_100.vtk
model=../../models/conv_autoencoder_scripted.pt

${anomap} modelTest \
${inp} \
${model} 256 64 float







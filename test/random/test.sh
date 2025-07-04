#!/bin/bash

echo "Running model test"

anomap=../../build/anomap

# inp=random_1K.vtk
inp=random_100.vtk
# model=../../models/conv_autoencoder_scripted.pt
model=../../models/lin10_float64_05_scripted_nonorm.pt
# model=../../models/lin10_40_scripted_nonorm.pt

${anomap} modelTest \
${inp} \
${model} 256 16 \
-c

# ${model} 256 64







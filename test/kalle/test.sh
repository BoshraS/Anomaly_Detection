#!/bin/bash

echo "Running model test"

anomap=../../build/anomap

inp=random_10K.vtk
#inp=random_100.vtk
# model=../../models/conv_autoencoder_scripted.pt
# model=../../models/lin10_float64_05_scripted_nonorm.pt
model=../../models/lin10_40_scripted_nonorm.pt
# model=../../models/lin10_float64_20_scripted_nonorm.pt
# model=../../models/lin10_float64_15_12_scripted_nonorm.pt

out1=enc_output.bin
out2=centers.clc
out3=scores.ano
out4=labels.clb

${anomap} encode -c -f \
${inp} \
${out1}


${anomap} findClusterCenters -c -f \
${out1} \
${out2} \
-d 100 \
-m ${model} 256 16 float 5.0


${anomap} score -c -f \
${out1} \
${out2} \
${out3} \
${out4}







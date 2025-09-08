#!/bin/bash

echo "Running model test"

anomap=../../build/anomap

inp8=../tractograms/100307_8_100k.vtk
inp16=../tractograms/100307_16_100k.vtk
inp32=../tractograms/100307_32_100k.vtk
inp64=../tractograms/100307_64_100k.vtk
inp128=../tractograms/100307_128_100k.vtk
inp256=../tractograms/100307_256_100k.vtk
inpoc=../tractograms/100307_original_100k.vtk


model64=../../models/conv_autoencoder_scripted.pt
#model32=../../models/asdf.pt
model16=../../models/lin10_40_scripted_nonorm.pt


echo "Starting model 64 - sample 8"
${anomap} modelTest -c \
${inp8} \
${model64} 8 64 float

echo "Starting model 64 - sample 16"
${anomap} modelTest -c \
${inp16} \
${model64} 16 64 float

echo "Starting model 64 - sample 32"
${anomap} modelTest -c \
${inp32} \
${model64} 32 64 float

echo "Starting model 64 - sample 64"
${anomap} modelTest -c \
${inp64} \
${model64} 64 64 float

echo "Starting model 64 - sample 128"
${anomap} modelTest -c \
${inp128} \
${model64} 128 64 float

echo "Starting model 64 - sample 256"
${anomap} modelTest -c \
${inp256} \
${model64} 256 64 float

echo "Starting model 64 - sample original"
${anomap} modelTest -c \
${inpoc} \
${model64} 256 64 float


echo "Starting model 16 - sample 8"
${anomap} modelTest -c \
${inp8} \
${model16} 8 16 float

echo "Starting model 16 - sample 16"
${anomap} modelTest -c \
${inp16} \
${model16} 16 16 float

echo "Starting model 16 - sample 32"
${anomap} modelTest -c \
${inp32} \
${model16} 32 16 float

echo "Starting model 16 - sample 64"
${anomap} modelTest -c \
${inp64} \
${model16} 64 16 float

echo "Starting model 16 - sample 128"
${anomap} modelTest -c \
${inp128} \
${model16} 128 16 float

echo "Starting model 16 - sample 256"
${anomap} modelTest -c \
${inp256} \
${model16} 256 16 float

echo "Starting model 16 - sample original"
${anomap} modelTest -c \
${inpoc} \
${model64} 256 16 float









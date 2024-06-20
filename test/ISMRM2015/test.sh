#!/bin/bash

echo "Running ISMRM 2015 challange data example"

anomap=../../build/anomap

inp=ISMRM2015_MNI

${anomap} encode -f \
${inp}.vtk \
${inp}.bin

${anomap} decode -f \
${inp}.bin \
${inp}_dec.vtk

${anomap} findClusterCenters -f \
${inp}.bin \
${inp}.clc \
--maxDist 1

${anomap} decode -f \
${inp}.clc \
${inp}_clc.vtk


${anomap} score -f \
${inp}.bin \
${inp}.clc \
${inp}.ano \
${inp}.clb

${anomap} toTrack -f \
${inp}.vtk \
-s ${inp}.ano \
-l ${inp}.clb \
${inp}_mapped.vtk

${anomap} toImg -f \
${inp}.vtk \
${inp}.ano \
${inp}_mapped.nii.gz \
--medFilt



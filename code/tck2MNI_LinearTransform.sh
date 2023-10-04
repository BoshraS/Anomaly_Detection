#!/bin/bash 

export ANTSPATH=/usr/local/ANTs/bin
for i in {S161,S198,S214,S216} ; do
	sub=/media/adt/Elements/Boshra/languagePatients4AE/${i}
	T1=${sub}/t1_LAS.nii.gz
	T1BET=${sub}/t1_BrainExtractionBrain_LAS.nii.gz
	MNI=/media/nas/AE_anomaly/MNI152_T1_1mm_brain.nii.gz
	T1MNI=${sub}/t1_LAS_MNI.nii.gz

	# linear rigid transformation
	antsRegistration \
	--dimensionality 3 \
	--float 0 \
	--interpolation Linear \
	--winsorize-image-intensities [0.1,0.9] \
	--metric Mattes[${T1BET},${MNI},1,128,Regular,0.25] \
	--transform Rigid[0.1] \
	--convergence [10000x10000x10000x10000x10000,1e-6,10] \
	--shrink-factors 4x3x3x2x2 \
	--smoothing-sigmas 32x16x8x2x1vox \
	--output ${sub}/antsRigid_

	antsApplyTransforms -d 3 -e 0 \
	-i ${T1} \
	-r ${MNI} \
	-o ${T1MNI} \
	-t [${sub}/antsRigid_0GenericAffine.mat,1] \
	-f 0
	# convert to binary file
	ConvertTransformFile \
	3 \
	${sub}/antsRigid_0GenericAffine.mat \
	${sub}/antsRigid_0GenericAffine.txt \
	--hm --ras
	# use transformation matrix to transform the tck to MNI space
	./nipt tractogram transform \
	-f \
	${sub}/11M_minlen60_N256.tck \
	${sub}/antsRigid_0GenericAffine.txt \
	${sub}/11M_minlen60_N256_MNI.tck
done

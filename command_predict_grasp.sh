#!/bin/bash
GPU_ID="0"

cd grasp_multiObject_multiGrasp/tools

CUDA_VISIBLE_DEVICES=${GPU_ID} python predict_graspRGD.py \
	--net res50 \
	--dataset grasp

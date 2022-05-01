#!/bin/sh
#BSUB -n 4
#BSUB -q gpu 
#BSUB -gpgpu 1
#BSUB -o logs/5.1/out.%J.log
#BSUB -e logs/5.1/err.%J.log

source /hpc/jhinno/unischeduler/exec/unisched

module load anaconda3			
module load cuda-11.1
source activate
conda activate nerf

cd /hpc/users/CONNECT/zipengwang/ShadowNeRF
python run.py
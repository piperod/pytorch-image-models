#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem=60GB
#SBATCH -J scale_454_alex_size_454
#SBATCH -o /users/xyu110/pytorch-image-models/scale_image_alexnet/scale_454_alex_size_454.out
#SBATCH -e /users/xyu110/pytorch-image-models/scale_image_alexnet/scale_454_alex_size_454.err
#SBATCH --account=carney-tserre-condo

#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL


module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate default

cd /users/xyu110/pytorch-image-models

sh distributed_val.sh 2 validate.py \
    --data-dir /gpfs/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC \
    --model alexnet \
    -b 128 \
    --image-scale 3 454 454 \
    --input-size 3 454 454 \
    --pretrained \
    --checkpoint /users/xyu110/pytorch-image-models/alexnet_size_454/last.pth.tar \

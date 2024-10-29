#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem=60GB
#SBATCH -J alex_size_454_scale_0.08
#SBATCH -o /users/xyu110/pytorch-image-models/alex_size_454_scale_0.08.out
#SBATCH -e /users/xyu110/pytorch-image-models/alex_size_454_scale_0.08.err
#SBATCH --account=carney-tserre-condo
#SBATCH --mail-user=xizheng_yu@brown.edu
#SBATCH --mail-type=END,FAIL


module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate default

cd /users/xyu110/pytorch-image-models

sh distributed_train.sh 2 train_skeleton.py \
    --data-dir /gpfs/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC \
    --model alexnet \
    --input-size 3 454 454 \
    --output /users/xyu110/scratch/alex_0.08 \
    --epochs 90 \
    --experiment alexnet_size_454_scale_0.08 \
    --opt sgd \
    -b 128 \
    --lr 1e-2 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --hflip 0.5\
    --train-crop-mode rrc\
    --scale 0.08 1.0

# resume training

# sh distributed_train.sh 2 train_skeleton.py \
#     --resume /users/xyu110/pytorch-image-models/alexnet_size_454/last.pth.tar \
#     --data-dir /gpfs/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC \
#     --model alexnet \
#     --input-size 3 454 454 \
#     --output /users/xyu110/pytorch-image-models \
#     --epochs 12 \
#     --experiment alexnet_size_454_resume \
#     --opt sgd \
#     -b 128 \
#     --lr 1e-2 \
#     --weight-decay 5e-4 \
#     --sched step \
#     --momentum 0.9 \
#     --lr-cycle-decay 0.1 \
#     --decay-epochs 30 \
#     --warmup-epochs 0 \
#     --hflip 0.5\
#     --train-crop-mode rrc\

# sh distributed_val.sh 2 validate.py \
#     --data-dir /gpfs/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC \
#     --model alexnet \
#     -b 128 \
#     --image-scale 3 454 454 \
#     --input-size 3 454 454 \
#     --pretrained \
#     --checkpoint /users/xyu110/pytorch-image-models/alexnet_size_454/last.pth.tar \
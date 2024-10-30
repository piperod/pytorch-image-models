#!/bin/bash
#SBATCH --time=99:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=80GB
#SBATCH -o hmax_ip_1.out
#SBATCH -e hmax_ip_1.err
#SBATCH --account=carney-tserre-condo
#SBATCH -J hmax_ip_1


module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate default

cd /users/xyu110/pytorch-image-models

sh distributed_train.sh 2 train_skeleton.py \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/ \
    --dataset torch/imagenet \
    --model chmax \
    --model-kwargs ip_scale_bands=1 classifier_input_size=4096 hmax_type='bypass'\
    --contrastive-loss \
    --opt sgd \
    -b 128 \
    --epochs 90 \
    --lr 1e-2 \
    --weight-decay 5e-4 \
    --sched step \
    --momentum 0.9 \
    --lr-cycle-decay 0.1 \
    --decay-epochs 30 \
    --warmup-epochs 0 \
    --hflip 0.5\
    --train-crop-mode rrc\
    --input-size 3 227 227\
    --experiment junk \
    --output /gpfs/data/tserre/xyu110/pytorch-output/train \
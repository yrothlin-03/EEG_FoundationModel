#!/bin/bash
#SBATCH -p H100,A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=pretraining_z512
#SBATCH --output=/home/infres/yrothlin-24/EEG_Foundation_Model/logs/pretraining/%x_%j.out
#SBATCH --error=/home/infres/yrothlin-24/EEG_Foundation_Model/logs/pretraining/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

eval "$(micromamba shell hook --shell=bash)"
micromamba activate rech202

cd /home/infres/yrothlin-24/EEG_Foundation_Model

mkdir -p logs/pretraining

python pretrain.py \
    --model_config configs/model.yaml \
    --pretraining_config configs/pretraining.yaml \
    --data_config configs/data.yaml \
    --training.num_epochs 20 \
    --training.out_dir /home/infres/yrothlin-24/EEG_Foundation_Model/outputs/pretraining/z_sweep/z512 \
    --training.max_train_step 5000 \
    --training.max_val_step 500 \
    --model.mode vq \
    --model.ch_mult "[1, 2, 4, 8]" \
    --model.z_channels 512 \
    --model.n_quantizers 8 \
    --model.channel_adaptor true

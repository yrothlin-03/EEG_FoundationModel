#!/bin/bash
#SBATCH -p H100,A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=contrastive_supcon
#SBATCH --output=/home/infres/yrothlin-24/EEG_Foundation_Model/logs/contrastive/%x_%j.out
#SBATCH --error=/home/infres/yrothlin-24/EEG_Foundation_Model/logs/contrastive/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

eval "$(micromamba shell hook --shell=bash)"
micromamba activate rech202

cd /home/infres/yrothlin-24/EEG_Foundation_Model

mkdir -p logs/contrastive

python contrastive.py \
    --model_config configs/model.yaml \
    --contrastive_config configs/contrastive.yaml \
    --data_config configs/data.yaml \
    --training.loss supcon \
    --training.num_epochs 50 \
    --training.out_dir /home/infres/yrothlin-24/EEG_Foundation_Model/outputs/contrastive/supcon \
    --training.max_step_train 300 \
    --training.max_step_val 100 \
    --supcon_datasets "[SLEEPEDF, BCI2A, SEEDV]" \
    --data.channel_mode mapped

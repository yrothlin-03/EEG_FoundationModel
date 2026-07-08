#!/bin/bash
#SBATCH -p H100,A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=downstream_sleepedfx
#SBATCH --output=/home/infres/yrothlin-24/EEG_Foundation_Model/logs/downstream/%x_%j.out
#SBATCH --error=/home/infres/yrothlin-24/EEG_Foundation_Model/logs/downstream/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

eval "$(micromamba shell hook --shell=bash)"
micromamba activate rech202

cd /home/infres/yrothlin-24/EEG_Foundation_Model

mkdir -p logs/downstream

python downstream.py \
    --model_config configs/model.yaml \
    --data_config configs/data.yaml \
    --downstream_config configs/downstream.yaml \
    --dataset BCI2A \
    --training.num_epochs 50 \
    --training.out_dir /home/infres/yrothlin-24/EEG_Foundation_Model/outputs/downstream/bci2a \
    --checkpoints_path /home/infres/yrothlin-24/EEG_Foundation_Model/outputs/pretraining/2_quantizers/pretraining_best.pt \

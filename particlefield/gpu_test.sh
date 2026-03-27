#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=graphic
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-10          # <-- update to 0-(N_VX*N_COUPLING-1)
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

cd "/data/biophys/schimmenti/Repositories/camilla/particlefield"

mkdir -p data2d logs

module load cuda/12.8
nvidia-smi

#!/bin/bash
#SBATCH --job-name=parfield_sweep
#SBATCH --partition=graphic
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-95
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

VX_VALUES=(0.05 0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.5)
COUPLING_VALUES=(0.001 0.01 0.05 0.1 0.5 1.0 1.5 2.0)

# ---------- modules ----------
module load cuda/12.8

cd "/data/biophys/schimmenti/Repositories/camilla/particlefield"

pythonapp="/home/schimmenti/miniconda3/bin/python"

# ---------- decode task ID into (vx, coupling) ----------
N_COUPLING=${#COUPLING_VALUES[@]}

i_vx=$(( SLURM_ARRAY_TASK_ID / N_COUPLING ))
i_coupling=$(( SLURM_ARRAY_TASK_ID % N_COUPLING ))

VX=${VX_VALUES[$i_vx]}
COUPLING=${COUPLING_VALUES[$i_coupling]}

echo "============================================"
echo "Array task : $SLURM_ARRAY_TASK_ID"
echo "vx         : $VX"
echo "coupling   : $COUPLING"
echo "Host       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "============================================"

mkdir -p data2d logs

# ---------- run ----------
$pythonapp particlefield2d.py \
    --vx       "$VX"       \
    --coupling "$COUPLING" \
    --dx       0.01        \
    --dy       0.01        \
    --Nx       16384       \
    --Ny       1024        \
    --dt       1e-3        \
    --field_length_scale  0.05  \
    --field_time_scale    1.0   \
    --particle_time_scale 0.1   \
    --mu       1.0         \
    --kappa    2.5         \
    --r0       1.0         \
    --vy       0.0         \
    --temp     0.004       \
    --sim_time 10000.0      \
    --num_snapshots 1000   \
    --seed     42          \
    --verbosity 1000

echo "Job finished with exit code $?"
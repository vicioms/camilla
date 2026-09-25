#!/bin/bash
#SBATCH --job-name=parfield_sweep
#SBATCH --partition=graphic
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-95
#SBATCH --chdir=/data/biophys/schimmenti/Repositories/camilla/particlefield
#SBATCH --output=/data/biophys/schimmenti/Repositories/camilla/particlefield/logs/job_%A_%a.out
#SBATCH --error=/data/biophys/schimmenti/Repositories/camilla/particlefield/logs/job_%A_%a.err

# Create logs/ in the project directory BEFORE calling sbatch.
# The corrected Python simulation must be saved as particlefield2d.py.
set -euo pipefail
trap 'job_status=$?; printf "Job finished with exit code %s\n" "$job_status"; exit "$job_status"' EXIT

VX_VALUES=(0.05 0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.5)
COUPLING_VALUES=(0.001 0.01 0.05 0.1 0.5 1.0 1.5 2.0)

# Override at submission, for example:
# sbatch --export=ALL,NUM_CYCLES=5,SNAPSHOT_STRIDE=4 parfield_sweep.sh
NUM_CYCLES="${NUM_CYCLES:-3}"
SNAPSHOT_STRIDE="${SNAPSHOT_STRIDE:-1}"
SEED=42

for value in "$NUM_CYCLES" "$SNAPSHOT_STRIDE"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf 'NUM_CYCLES and SNAPSHOT_STRIDE must be positive integers.\n' >&2
        exit 2
    fi
done

task_id="${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch as a job array}"
array_job_id="${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID}"
n_coupling=${#COUPLING_VALUES[@]}
n_tasks=$(( ${#VX_VALUES[@]} * n_coupling ))

if [[ ! "$task_id" =~ ^[0-9]+$ ]]; then
    printf 'Invalid array task ID: %s\n' "$task_id" >&2
    exit 2
fi
task_id=$((10#$task_id))
if (( task_id >= n_tasks )); then
    printf 'Array task ID %s is outside 0-%s.\n' "$task_id" "$((n_tasks - 1))" >&2
    exit 2
fi

i_vx=$((task_id / n_coupling))
i_coupling=$((task_id % n_coupling))
VX="${VX_VALUES[$i_vx]}"
COUPLING="${COUPLING_VALUES[$i_coupling]}"

module load cuda/12.8

PROJECT_DIR="/data/biophys/schimmenti/Repositories/camilla/particlefield"
pythonapp="/home/schimmenti/miniconda3/bin/python"
cd -- "$PROJECT_DIR"

if [[ ! -x "$pythonapp" || ! -f particlefield2d.py ]]; then
    printf 'Missing Python interpreter or particlefield2d.py.\n' >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

# All tasks share a sweep parent. The updated Python script constructs the
# complete parameter hierarchy, configuration fingerprint, and unique run leaf.
# Repeated or concurrent runs therefore get separate directories automatically.
output_root="$PROJECT_DIR/data2d/sweep_${array_job_id}"
mkdir -p -- "$output_root"

printf 'Array task: %s / %s\n' "$task_id" "$((n_tasks - 1))"
printf 'vx=%s coupling=%s cycles=%s seed=%s\n' "$VX" "$COUPLING" "$NUM_CYCLES" "$SEED"
printf 'Host: %s\nOutput root: %s\n' "${HOSTNAME:-unknown}" "$output_root"
# Python prints the complete final run directory in this task's log.

# With vy=0, one cycle is Lx/abs(vx) = 40.96/abs(vx).
# Python shortens the last step to finish at exactly NUM_CYCLES cycles.
# srun's exit code is propagated to Slurm by set -e and the EXIT trap.
srun --ntasks=1 --cpus-per-task="$OMP_NUM_THREADS" \
    "$pythonapp" -u particlefield2d.py \
    --device cuda \
    --vx "$VX" \
    --vy 0.0 \
    --coupling "$COUPLING" \
    --dx 0.01 \
    --dy 0.01 \
    --Nx 4096 \
    --Ny 1024 \
    --dt 1e-3 \
    --field_length_scale 1.0 \
    --field_time_scale 1.0 \
    --particle_time_scale 0.1 \
    --mu 1.0 \
    --kappa 2.5 \
    --r0 1.0 \
    --temp 0.004 \
    --num_cycles "$NUM_CYCLES" \
    --num_snapshots 100 \
    --snapshot_stride "$SNAPSHOT_STRIDE" \
    --output_dir "$output_root" \
    --seed "$SEED" \
    --verbosity 1000

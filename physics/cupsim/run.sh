#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=graphic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500M
#SBATCH --mail-type=END
#SBATCH --job-name="GPU TEST"
#SBATCH --output=/home/schimmenti/output.txt
#SBATCH --error=/home/schimmenti/error.txt
#SBATCH --gres=gpu:1

#module load cuda/12.1
#nvidia-smi
#nvcc --version

project="/data/biophys/schimmenti/Repositories/camilla/cupsim"
scratch="/scratch/schimmenti/"

mkdir -p $scratch || exit 1
cd $scratch || exit 1
cp $project/main.cu . || exit 1
cp $project/simulation.cuh . || exit 1
cp $project/basics.cuh . || exit 1
cp $project/cudaKernels.cuh . || exit 1
cp $project/helper_math.h . || exit 1
module load cuda/12.1
nvcc main.cu -o main.o -O3
#cp $project/main.o . || exit 1
srun ./main.o
cp particles.txt $project/particles.txt
# Clean up
cd || exit 1
rm -rf $scratch || exit 1
exit 0
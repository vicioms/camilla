#!/bin/bash
VX_VALUES=(0.1 0.5 1.0 1.5 2.0)
COUPLING_VALUES=(0.01 0.05 0.1 0.5 1.0)

# ---------- setup ----------
cd "/data/biophys/schimmenti/Repositories/camilla/particlefield"
pythonapp="/home/schimmenti/miniconda3/bin/python"

mkdir -p data2d logs

N_VX=${#VX_VALUES[@]}
N_COUPLING=${#COUPLING_VALUES[@]}
TOTAL=$(( N_VX * N_COUPLING ))
JOB=0

for i_vx in $(seq 0 $(( N_VX - 1 ))); do
  for i_coupling in $(seq 0 $(( N_COUPLING - 1 ))); do
    JOB=$(( JOB + 1 ))
    VX=${VX_VALUES[$i_vx]}
    COUPLING=${COUPLING_VALUES[$i_coupling]}

    echo "[$JOB/$TOTAL] vx=$VX  coupling=$COUPLING"

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
      --sim_time 1000.0      \
      --num_snapshots 1000   \
      --seed     42          \
      --verbosity 1000

    echo "  -> exit code $?"
  done
done

echo "All $TOTAL jobs done."
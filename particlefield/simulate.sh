#!/usr/bin/env bash
set -euo pipefail

# Run the particle-field simulation

python_app="$HOME/anaconda3/envs/organoids/bin/python"

dx=0.01
N=4096
dt=1e-4

max_jobs=8
running_jobs=0

field_length_scale_start=0.03
field_length_scale_factor=5
num_field_length_scales=1

field_time_scale_start=0.1
field_time_scale_factor=5
num_field_time_scales=1

particle_time_scale_start=0.001
particle_time_scale_factor=5
num_particle_time_scales=1

mu_start=1.0
mu_factor=5
num_mu_values=1

kappa_start=50.0
kappa_factor=1
num_kappa_values=1

coupling_start=0.01
coupling_end=10
num_coupling_values=5
coupling_factor=$(awk "BEGIN{print ($coupling_end/$coupling_start)^(1/($num_coupling_values-1))}")

r0_start=2.0
r0_factor=1
num_r0_values=1

v_start=0.002
v_end=2.0
num_v_values=5
v_factor=$(awk "BEGIN{print ($v_end/$v_start)^(1/($num_v_values-1))}")

temperature_param=0.004

num_cycles=3

for i in $(seq 0 $((num_field_length_scales-1))); do
  for j in $(seq 0 $((num_field_time_scales-1))); do
    for k in $(seq 0 $((num_particle_time_scales-1))); do
      for l in $(seq 0 $((num_mu_values-1))); do
        for m in $(seq 0 $((num_kappa_values-1))); do
          for n in $(seq 0 $((num_coupling_values-1))); do
            for o in $(seq 0 $((num_r0_values-1))); do
              for p in $(seq 0 $((num_v_values-1))); do

                field_length_scale=$(echo "$field_length_scale_start * ($field_length_scale_factor ^ $i)" | bc -l)
                field_time_scale=$(echo "$field_time_scale_start * ($field_time_scale_factor ^ $j)" | bc -l)
                particle_time_scale=$(echo "$particle_time_scale_start * ($particle_time_scale_factor ^ $k)" | bc -l)
                mu=$(echo "$mu_start * ($mu_factor ^ $l)" | bc -l)
                kappa=$(echo "$kappa_start * ($kappa_factor ^ $m)" | bc -l)
                coupling=$(echo "$coupling_start * ($coupling_factor ^ $n)" | bc -l)
                r0=$(echo "$r0_start * ($r0_factor ^ $o)" | bc -l)
                v=$(echo "$v_start * ($v_factor ^ $p)" | bc -l)

                echo "Running: fls=$field_length_scale fts=$field_time_scale pts=$particle_time_scale mu=$mu kappa=$kappa coupling=$coupling r0=$r0 v=$v"

                "$python_app" particlefield.py \
                  --dx "$dx" --N "$N" --dt "$dt" \
                  --field_length_scale "$field_length_scale" \
                  --field_time_scale "$field_time_scale" \
                  --particle_time_scale "$particle_time_scale" \
                  --mu "$mu" --kappa "$kappa" \
                  --coupling "$coupling" --r0 "$r0" --v "$v" \
                  --temp "$temperature_param" --num_cycles "$num_cycles" &

                ((running_jobs+=1))
                if (( running_jobs >= max_jobs )); then
                  wait -n
                  ((running_jobs-=1))
                fi

              done
            done
          done
        done
      done
    done
  done
done

wait
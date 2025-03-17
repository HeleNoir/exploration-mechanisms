#!/usr/bin/env bash

#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --nodelist=oc226
#SBATCH --cpus-per-task=25
#SBATCH --mem=25G
#SBATCH --array=1-24
#SBATCH --output=/data/oc226/%u/output/output-%A_%a.txt

function_param=$SLURM_ARRAY_TASK_ID

srun nix develop --impure --command cargo run --release --package exploration-mechanisms --bin pso_rr -- --function $function_param --dimensions $dim
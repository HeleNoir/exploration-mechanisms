#!/usr/bin/env bash

export dim=$1

sbatch slurm/pso.sbatch
sbatch slurm/shade.sbatch
sbatch slurm/pso_rr.sbatch
sbatch slurm/pso_gpgm.sbatch
sbatch slurm/pso_npgm.sbatch
sbatch slurm/pso_pdm.sbatch
sbatch slurm/pso_srm.sbatch

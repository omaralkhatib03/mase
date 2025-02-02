#!/bin/bash
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=2:gpu_type=RTX6000

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate mase_env
which python
cd /rds/general/user/oa321/home/mase/tasks/tutorial5/
python task2a.py


#!/bin/bash
#SBATCH -o /hpcgpfs01/scratch/akumar/code/cpd/protein_data/protein_1fme/SLURM/OUTPUT/random_4.out
#SBATCH -p volta
#SBATCH -t 02:30:00
#SBATCH --gres=gpu:1
#SBATCH -A student-v
#SBATCH -J 1fme_r_4

source ~/.bashrc
python $PROJ_DIR_PATH/cpd/protein_data/protein_1fme/random_4.py
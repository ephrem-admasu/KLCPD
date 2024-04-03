#!/bin/bash
#SBATCH -o /hpcgpfs01/scratch/akumar/code/cpd/protein_data/protein_1fme/SLURM/OUTPUT/svds_3.out
#SBATCH -p volta
#SBATCH -t 02:30:00
#SBATCH --gres=gpu:1
#SBATCH -A student-v
#SBATCH -J 1fme_s_3

source ~/.bashrc
python $PROJ_DIR_PATH/cpd/protein_data/protein_1fme/svds_3.py
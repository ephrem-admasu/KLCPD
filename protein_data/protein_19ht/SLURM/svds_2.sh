#!/bin/bash
#SBATCH -o /hpcgpfs01/scratch/akumar/code/cpd/protein_data/protein_19ht/SLURM/OUTPUT/svds_2.out
#SBATCH -p volta
#SBATCH -t 02:30:00
#SBATCH --gres=gpu:1
#SBATCH -A student-v
#SBATCH -J 19ht_s_2

source ~/.bashrc
python $PROJ_DIR_PATH/cpd/protein_data/protein_19ht/svds_2.py
#!/bin/sh
#SBATCH -o /hpcgpfs01/scratch/akumar/code/cpd/protein_data/protein_1fme/SLURM/slurm_jobs.out
#SBATCH -p volta
#SBATCH -t 02:30:00
#SBATCH --gres=gpu:1
#SBATCH -A student-v
#SBATCH -J slurm_jobs

source ~/.bashrc
sbatch $PROJ_DIR_PATH/cpd/protein_data/protein_1fme/SLURM/random_2.sh

sbatch $PROJ_DIR_PATH/cpd/protein_data/protein_1fme/SLURM/random_3.sh

sbatch $PROJ_DIR_PATH/cpd/protein_data/protein_1fme/SLURM/random_4.sh

sbatch $PROJ_DIR_PATH/cpd/protein_data/protein_1fme/SLURM/svds_2.sh

sbatch $PROJ_DIR_PATH/cpd/protein_data/protein_1fme/SLURM/svds_3.sh
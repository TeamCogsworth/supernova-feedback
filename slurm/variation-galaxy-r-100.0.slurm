#!/bin/bash
## Job Name
#SBATCH --job-name=r-100.0
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=2:00:00
#SBATCH -o /mnt/home/twagg/supernova-feedback/slurm/logs/r-100.0_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tomwagg@uw.edu
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

echo "r-100.0"

python /mnt/home/twagg/supernova-feedback/src/variations-galaxy.py -f "r-100.0.h5" -r 100.0
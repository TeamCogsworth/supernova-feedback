#!/bin/bash
## Job Name
#SBATCH --job-name=half_Z
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=1:00:00
#SBATCH -o /mnt/home/twagg/supernova-feedback/slurm/logs_%a_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tomwagg@uw.edu
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

echo "m11h half Z"

python /mnt/home/twagg/supernova-feedback/src/lower_metallicity.py

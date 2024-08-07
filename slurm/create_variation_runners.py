basic = """#!/bin/bash
## Job Name
#SBATCH --job-name=NAMEHERE
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=1:30:00
#SBATCH -o /mnt/home/twagg/supernova-feedback/slurm/logs/NAMEHERE_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tomwagg@uw.edu
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

echo "NAMEHERE"

python /mnt/home/twagg/supernova-feedback/src/variations.py -f "NAMEHERE.h5" -FLAGHERE VALHERE"""

for name, flag, value in zip(["ce-0.1", "ce-10.0", "beta-0.0", "beta-0.5", "beta-1.0", "ecsn-265", "no-fallback"],
                             ['c', 'c', 'b', 'b', 'b', 'e', 'k'],
                             ['0.1', '10.0', '0.0', '0.5', '1.0', '-265', '3']):
    with open(f"variation-{name}.slurm", "w") as f:
        write_this = basic.replace("NAMEHERE", name).replace("FLAGHERE", flag).replace("VALHERE", str(value))
        f.write(write_this)

#!/bin/bash
#SBATCH --time=0:55:00
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH --array=0-1000
#SBATCH --partition=packable
#SBATCH --account=j1042
#SBATCH --job-name=LLAGN
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3

# Need to delay job start times by random number to prevent overloading system:
sleep `expr $RANDOM % 60`


# The fermipy  environment first needs to be sourced:
source /zfs/astrohe/Software/fermipy_1.2.sh

#Change to working directory and run job
cd $SLURM_SUBMIT_DIR
python submit_fermi_stacking_jobs.py $SLURM_ARRAY_TASK_ID


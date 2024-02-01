#PBS -N array
#PBS -o palmetto_output/%s_%s%s_out.txt
#PBS -e palmetto_output/%s_%s%s_err.txt
#PBS -l select=1:ncpus=1:mem=15gb:interconnect=1g,walltime=25:00:00
#PBS -J 0-100

#Need to delay job start times by random number to prevent overloading system:
sleep `expr $RANDOM % 60`

# source environment
source /zfs/astrohe/Software/fermipy_1.2.sh
python submit_fermi_stacking_jobs.py $PBS_ARRAY_INDEX 










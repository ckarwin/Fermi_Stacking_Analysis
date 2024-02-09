# Imports:
import pandas as pd
import os, sys
import time 
from astropy.io import fits
import yaml

# Load sample data from yaml file:
with open("inputs.yaml","r") as file:
    inputs = yaml.load(file,Loader=yaml.FullLoader)

this_file = inputs["sample_file"]
file_type = inputs["file_type"]
column_name = inputs["column_name"]
column_ra = inputs["column_ra"]
column_dec = inputs["column_dec"]
run_list = inputs["run_list"]
psf_low = inputs["psf_low"]
psf_high = inputs["psf_high"]
run_name = inputs["run_name"]
job_type = inputs["job_type"]

if file_type == "fits":
    hdu = fits.open(this_file)
    data = hdu[1].data
    name_list = data[column_name].strip().tolist()
    ra_list = data[column_ra].tolist()
    dec_list = data[column_dec].tolist()

if file_type == "csv":
    df = pd.read_csv(this_file)
    name_list = df[column_name].tolist()
    ra_list = df[column_ra].tolist()
    dec_list = df[column_dec].tolist()

if file_type == "tab":
    df = pd.read_csv(this_file, delim_whitespace=True)
    name_list = df[column_name].tolist()
    ra_list = df[column_ra].tolist()
    dec_list = df[column_dec].tolist()

# Specify which sources to run.
# Set to name_list for full sample.
if run_list == "default":
    print("ERROR: rerunning full sample!")
    sys.exit()

# Get current working directory:
this_dir = os.getcwd()

# Make output directory:
if(os.path.isdir("palmetto_output")==False):
    os.system('mkdir palmetto_output')

# Construct PSF list:
psf_list = []
for i in range(psf_low,psf_high):
    psf_list += [i]*len(name_list)

# Duplicate lists to match PSF runs:
name_list = name_list*psf_high
ra_list = ra_list*psf_high
dec_list = dec_list*psf_high


for i in range(len(name_list)):

    # Specify run:
    this_name = name_list[i]
    this_ra = ra_list[i]
    this_dec = dec_list[i]
    this_psf = psf_list[i]

    if this_name in run_list:
        print()
        print("running %s..." %this_name)
        f = open("rerun.pbs","w")
        f.write("#PBS -N %s\n" %this_name)
        f.write("#PBS -l select=1:ncpus=2:mem=125gb:interconnect=1g,walltime=90:00:00\n\n")
        f.write("source /zfs/astrohe/Software/lat-stacking.sh\n")
        f.write("cd %s\n" %this_dir)
        f.write("python client.py '%s' %s %s %s" % (this_name,this_ra,this_dec,this_psf))
        f.close()
        os.system("qsub rerun.pbs")
        time.sleep(3)

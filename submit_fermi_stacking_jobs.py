# Imports:
import pandas as pd
import os, sys
import time 
from astropy.io import fits
import yaml

# Load sample data from yaml file:
with open(input_yaml,"r") as file:
    inputs = yaml.load(file)

this_file = inputs["sample_file"]
file_type = inputs["file_type"]
column_name = inputs["column_name"]
column_ra = inputs["column_ra"]
column_dec = inputs["column_dec"]
run_list = inputs["run_list"]
psf_low = inputs["psf_low"]
psf_high = inputs["psf_high"]
run_name = inputs["run_name"]
resource = inputs["resource"]

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

# Specify which sources to run.
# Set to name_list for full sample.
if run_list == "default":
    run_list = name_list

# Get current working directory:
this_dir = os.getcwd()

# Submit jobs:
for j in range(psf_low,psf_high): # PSF iterator for JLA (use 0 for standard analysis).
    for i in range(0,len(name_list)):

	this_name = name_list[i]
	this_ra = ra_list[i]
	this_dec = dec_list[i]

        if this_name in run_list:

    	    "Submitting source: " + str(this_name)

	    f = open('multiple_batch_submission.pbs','w')

	    f.write("#PBS -N %s_%s\n" %(run_name,str(this_name)))
            f.write("#PBS -l select=1:%s\n\n" %resource)
	    f.write("#the Fermi environment first needs to be sourced:\n")
	    f.write("cd /zfs/astrohe/Software\n")
	    f.write("source fermi.sh\n\n")
	    f.write("#change to working directory and run job\n")
	    f.write("cd %s\n" %this_dir)
	    f.write("python client.py '%s' %s %s %s" % (this_name,this_ra,this_dec,j))
	    f.close()
	
	    os.system("qsub multiple_batch_submission.pbs")
	    
            # Sleep a bit in order to not overwhelm the batch system:
            time.sleep(3)

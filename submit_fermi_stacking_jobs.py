# Imports:
import pandas as pd
import os, sys
import time 
from astropy.io import fits

# Sample data:
this_file = "/zfs/astrohe/ckarwin/Stacking_Analysis/LLAGN/Sample/LLAGN_control_sample_full.fits"
hdu = fits.open(this_file)
data = hdu[1].data
name_list = data["Name_1"].strip().tolist()
ra_list = data["_RAJ20001"].tolist()
dec_list = data["_DEJ20001"].tolist()

# Specify which source to run.
# Set to name_list for full sample.
run_list = ["NGC_221"]

# Get current working directory:
this_dir = os.getcwd()

# Submit jobs:
for j in range(0,1): # PSF iterator for JLA (use 0 for standard analysis).
    for i in range(0,len(name_list)):

	this_name = name_list[i]
	this_ra = ra_list[i]
	this_dec = dec_list[i]

        if this_name in run_list:

    	    "Submitting source: " + str(this_name)

	    f = open('multiple_batch_submission.pbs','w')

	    f.write("#PBS -N LLAGN_%s\n" %str(this_name))
            f.write("#PBS -l select=1:ncpus=2:mem=45gb:interconnect=1g,walltime=72:00:00\n\n")
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

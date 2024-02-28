# Imports:
import pandas as pd
import os, sys
import time 
from astropy.io import fits
import yaml
import numpy as np

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
submission_type = inputs["submission_type"]

# Specify submission type:
if submission_type == "array":
    this_run = int(sys.argv[1])
if submission_type == "array-parallel":
    this_run = int(sys.argv[1]) + int(sys.argv[2])

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

# Specify which sources to run:
if run_list != "default":
    name_list = np.array(name_list)
    ra_list = np.array(ra_list)
    dec_list = np.array(dec_list)
    run_index = np.isin(name_list,run_list)
    name_list = name_list[run_index].tolist()
    ra_list = ra_list[run_index].tolist()
    dec_list = dec_list[run_index].tolist()

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

# Specify run:
this_name = name_list[this_run]
this_ra = ra_list[this_run]
this_dec = dec_list[this_run]
this_psf = psf_list[this_run]

# Run job:
os.system("python client.py '%s' %s %s %s" % (this_name,this_ra,this_dec,this_psf))

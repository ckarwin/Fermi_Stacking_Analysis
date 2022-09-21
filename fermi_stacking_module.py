############################################################
# 
# Written by Chris karwin; April 2022; Clemson University.
#
# Based on original code from Marco Ajello, Vaidehi Paliya, and Abhishek Desai.
#
# Purpose: Main script for Fermi-LAT stacking analysis.
# 
# Index of functions:
#
#   Stacking(superclass)
#       - ang_sep(ra0, dec0, ra1, dec1)
#       - run_preprocessing(srcname,ra,dec)
#       - make_preprocessing_summary()
#       - PL2(Fit,name)
#       - run_stacking(srcname,PSF,indir="default")
#       - combine_likelihood(exclusion_list, savefile)
#       - plot_final_array(savefig,array)
#       - make_butterfly(name)
#       - get_UL95(array_file, ul_index=2.0)
#       - calc_upper_limit(srcname,ul_emin,ul_emax,comp_list=[0,1,2,3],mult_lt=False)
#
###########################################################

########################################
# Imports
from time import sleep
from random import randint
import resource
import random,shutil,yaml
import os,sys
from math import *
import numpy as np
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from fermipy.gtanalysis import GTAnalysis
import gc
import pyLikelihood 
from BinnedAnalysis import *
from astropy.io import fits
import pandas as pd
import math
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy.ndimage.filters import gaussian_filter
from astropy.convolution import convolve, Gaussian2DKernel
from matplotlib.cm import register_cmap,cmap_d
from scipy import stats, interpolate, optimize
from matplotlib import ticker, cm
from scipy.stats.contingency import margins
import PopStudies as PS
from IntegralUpperLimit import calc_int
from UpperLimits import UpperLimits
from SummedLikelihood import *
from astropy.io import fits
#######################################

# Superclass:
class StackingAnalysis:
   
    """Main inputs are specified in inputs.yaml file"""

    def __init__(self,input_yaml):

        # Get home directory:
        self.home = os.getcwd()
	
        # Load main inputs from yaml file:
        with open(input_yaml,"r") as file:
            inputs = yaml.load(file)

        # Main default inputs:
        self.ft1 = inputs["ft1"] # data file
        self.ft2 = inputs["ft2"] # spacecraft file
        self.galdiff = inputs["galdiff"] # Galactic diffuse model	
        self.isodiff = inputs["isodiff"] # isotropic model

        # ltcube:
	self.ltcube = inputs["ltcube"]
   
        # Scratch directory:
        self.use_scratch = inputs["use_scratch"]
        if self.use_scratch == True:
            self.scratch = inputs["scratch"]
            if(os.path.isdir(self.scratch)==False):
                os.system('mkdir %s' %self.scratch)
      
        # Perform joint likelihood analysis (True) or standard analysis (False):
        self.JLA = inputs["JLA"]
        
        # Main analysis parameters:
        self.emin = inputs["emin"]
        self.emax = inputs["emax"]
        self.tmin = inputs["tmin"]
        self.tmax = inputs["tmax"]
        self.zmax = inputs["zmax"]
        self.index_min = inputs["index_min"]
        self.index_max = inputs["index_max"]
        self.flux_min = inputs["flux_min"]
        self.flux_max = inputs["flux_max"]
        self.num_flux_bins = inputs["num_flux_bins"]

        # Option to run 4FGL sources:
        # Note: If True, must provide "remove_list.csv" file in run directory, with col1=sample_name, col2=4fgl_name, and no header.
        self.delete_4fgl = inputs["delete_4fgl"]
        if self.delete_4fgl == True:
            df = pd.read_csv("remove_list.csv",names=["col0","col1"])
            self.delete_sample_name = np.array(df["col0"].tolist())
            self.delete_4fgl_name = np.array(df["col1"].tolist())
        
        # Additional options:
        self.show_plots = inputs["show_plots"]
        self.calc_sed = inputs["calc_sed"]
        self.logEbins = inputs["sed_logEbins"]

        # Sample data:
        self.sample_file = inputs["sample_file"]
        self.file_type = inputs["file_type"]
        self.column_name = inputs["column_name"]
        
        if self.file_type == "fits":
            hdu = fits.open(self.sample_file)
            data = hdu[1].data
            self.sample_name_list = data[self.column_name].strip().tolist()

        if self.file_type == "csv":
            df = pd.read_csv(self.sample_file)
            self.sample_name_list = df[self.column_name].tolist()

    ################
    # Preprocessing:

    def ang_sep(self,ra0, dec0, ra1, dec1):
       
        """
        
        Calculate angular distance between two points on the sky.

        Inputs:
            - ra0, dec0: coordinates of point 1
            - ra1, dec1: coordinates of point 2 

        """
        
        C = np.pi/180.
        d0 = C * dec0
        d1 = C * dec1
        r12 = C * (ra0 - ra1)
        cd0 = np.cos(d0)
        sd0 = np.sin(d0)
        cd1 = np.cos(d1)
        sd1 = np.sin(d1)
        cr12 = np.cos(r12)
        sr12 = np.sin(r12)
        num = np.sqrt((cd0 * sr12) ** 2 + (cd1 * sd0 - sd1 * cd0 * cr12) ** 2)
        den = sd0 * sd1 + cd0 * cd1 * cr12
        
        return np.arctan2(num, den) / C

    def run_preprocessing(self,srcname,ra,dec):
       
        """
        
        Perform preprocessing of sources.

        Inputs:
            - srcname: name of source (string)
            - ra: right ascension (float)
            - dec: declination (float)

        """

	# Make print statement:
	print
	print "********** Fermi Stacking Analysis **********"
	print "Running preprocessing..."
	print
 
        # Need to specify coordinates as type float
        ra = float(ra)
        dec = float(dec)

        # Define main output directory for source:
        preprocess_output = os.path.join(self.home,"Preprocessed_Sources")
        if(os.path.isdir(preprocess_output)==False):
            os.system('mkdir %s' %preprocess_output)

        # Remove src output directory if it already exists:
        src_output_main = os.path.join(preprocess_output,srcname)
        if(os.path.isdir(src_output_main)==True):
            shutil.rmtree(src_output_main)

        # Define scratch directory for source:
        if self.use_scratch == True:
            preprocess_scratch = os.path.join(self.scratch,"Preprocessing")
            if(os.path.isdir(preprocess_scratch)==False):
                os.system('mkdir %s' %preprocess_scratch)
            src_scratch = os.path.join(preprocess_scratch,srcname)
            if(os.path.isdir(src_scratch)==True):
                shutil.rmtree(src_scratch)
            os.system('mkdir %s' %src_scratch)
            os.chdir(src_scratch)

        if self.use_scratch == False:
            os.chdir(src_output_main)
        	
	with open('%s.yaml' % srcname, 'w') as yml:
		yml.write("logging:\n")
		yml.write("  verbosity : 3\n")
		yml.write("  chatter : 3\n")
		yml.write("#--------#\n")
		yml.write("fileio:\n")
		yml.write("  outdir : output\n")
		yml.write("  logfile : %s\n" %srcname)
		yml.write("  usescratch : False\n")
		yml.write("  scratchdir : scratch\n")
		yml.write("#--------#\n")
		yml.write("data:\n")
		yml.write("  evfile : '%s'\n" %self.ft1)
		yml.write("  scfile : '%s'\n" %self.ft2)
                yml.write("  ltcube : '%s'\n" %self.ltcube)
		yml.write("#--------#\n")
		yml.write("binning:\n")
		yml.write("  roiwidth : 10\n")
		yml.write("  binsz : 0.08\n")
		yml.write("  binsperdec : 8\n")
		yml.write("#--------#\n")
		yml.write("selection:\n")
                yml.write("  emin : %s\n" %self.emin)
                yml.write("  emax : %s\n" %self.emax)
                yml.write("  zmax : %s\n" %self.zmax)
		yml.write("  target : '%s'\n" %srcname)
		yml.write("  radius : 15\n")
                yml.write("  tmin : %s\n" %self.tmin)
                yml.write("  tmax : %s\n" %self.tmax)
                yml.write("  evclass : 128\n")
                yml.write("  evtype : 3\n")
                yml.write("  filter : 'DATA_QUAL>0 && LAT_CONFIG==1'\n")
		yml.write("#--------#\n")
		yml.write("gtlike:\n")
		yml.write("  edisp : True\n")
		yml.write("  edisp_disable : ['isodiff']\n")
		yml.write("  irfs : 'P8R3_SOURCE_V2'\n")
		yml.write("#--------#\n")
		yml.write("model:\n")
		yml.write("  src_radius : 15\n")
		yml.write("  src_roiwidth : 15\n")
		yml.write("  galdiff : '%s'\n" %self.galdiff)
                yml.write("  isodiff : '%s'\n" %self.isodiff)
		yml.write("  catalogs :\n")
		yml.write("    - '4FGL'\n")
                yml.write("  extdir : '/zfs/astrohe/ckarwin/Stacking_Analysis/UFOs/Extended_Source_Archives/Extended_archive_v18/'\n")
		yml.write("  sources :\n")
		yml.write("    - { 'name' : '%s', 'ra' : %s, 'dec' : %s, 'SpectrumType' : PowerLaw }\n" %(srcname,ra,dec))
                yml.write("#--------#\n")
                yml.write("plotting:\n")
		yml.write("  format : png\n")
		yml.write("#--------#\n")
		yml.write("sed:\n")
		yml.write("  use_local_index : True\n")
	        
                # Include components for joint likelihood analysis (JLA):
                if self.JLA == True:
                    yml.write("components:\n")
                    yml.write("  - { model: {isodiff: iso_P8R3_SOURCE_V2_PSF0_v1.txt},\n")
                    yml.write("      selection : { evtype : 4 } }\n")
                    yml.write("  - { model: {isodiff: iso_P8R3_SOURCE_V2_PSF1_v1.txt},\n")
                    yml.write("      selection : { evtype : 8 } }\n")
                    yml.write("  - { model: {isodiff: iso_P8R3_SOURCE_V2_PSF2_v1.txt},\n")
                    yml.write("      selection : { evtype : 16 } }\n")
                    yml.write("  - { model: {isodiff: iso_P8R3_SOURCE_V2_PSF3_v1.txt},\n")
                    yml.write("      selection : { evtype : 32 } }\n")
        
        yml.close()
	
        gta = GTAnalysis('%s.yaml' % srcname,logging={'verbosity' : 3})
        gta.setup()
    
        # if rerunning 4fgl source, first delete source from model:
        if self.delete_4fgl == True:
            delete_sample_name = self.delete_sample_name
            delete_4fgl_name = self.delete_4fgl_name
            if srcname in delete_sample_name:
                this_index = srcname == delete_sample_name
                this_name = delete_4fgl_name[this_index][0]
                gta.delete_source(this_name)

	gta.optimize()
        gta.print_roi()
        gta.free_source('galdiff')
        gta.free_source('isodiff')
	gta.free_sources(minmax_ts=[25,None],pars='norm', distance=5.0)
        gta.free_sources(minmax_ts=[500,None],distance=7.0)
        gta.free_source(srcname)
        gta.fit()
        gta.write_roi('fit_model_1')
	gta.delete_source(srcname)
	model2 = {'Index' : 2.0, 'SpatialModel' : 'PointSource'}
	finder2 = gta.find_sources(prefix='find2',model=model2,sqrt_ts_threshold=4.0,
				min_separation=0.5,max_iter=10,sources_per_iter=20,
				tsmap_fitter='tsmap')
        gta.write_roi('fit_model_2')

	names2    = np.array([finder2['sources'][i]['name'] for i in range(len(finder2['sources']))])
	ra_ns2    = np.array([finder2['sources'][i]['ra'] for i in range(len(finder2['sources']))])
	dec_ns2   = np.array([finder2['sources'][i]['dec'] for i in range(len(finder2['sources']))])
	r95_ns2   = np.array([finder2['sources'][i]['pos_r95'] for i in range(len(finder2['sources']))])
	dist_sep = '%.2f' %0
	namee=''
	
	# Check ang separation if more than one source is found close to ra and dec
	if len(names2)>0:
		for i in range(len(names2)):
			sepp2=self.ang_sep(ra_ns2[i],dec_ns2[i],ra,dec)
			if sepp2 < r95_ns2[i] and sepp2 < 0.2:
				print names2[i],sepp2,r95_ns2[i]
				dist_sep = '%.2f' %sepp2
				namee = names2[i]
	if namee:                
                
                # Write name of replaced source:
                f = open("output/replaced_source_name.txt","w")
                f.write(str([srcname,namee]))
                f.close()

                srcname=namee
	else:
        	gta.add_source(srcname,{ 'ra' : ra, 'dec' : dec,
        		'SpectrumType' : 'PowerLaw', 'Index' : 2.0,
        		'Scale' : 1000, 'Prefactor' : 1e-11,
        		'SpatialModel' : 'PointSource' })
	gta.optimize()
	gta.print_roi()
        gta.free_source('galdiff')
        gta.free_sources(minmax_ts=[500,None],distance=7.0)
        gta.free_source(srcname)
	gta.fit()
        gta.write_roi('fit_model_3')
        
        # Calculate SED:
        if self.calc_sed == True:
            gta.sed(srcname,loge_bins=self.logEbins)

        p = np.load('output/fit_model_3.npy').flat[0]
        src = p['sources'][srcname]
	if str(src['ts'])=='nan':	# To check non-convergence
		print '****************************'
		print 'Fit has not converged'
		print '****************************'
		gta.free_sources(minmax_ts=[None,100],free=False)
		gta.free_source(srcname)
		gta.fit(tol=1e-8)
		gta.write_roi('fit_model_3')
		p = np.load('output/fit_model_3.npy').flat[0]
        	src = p['sources'][srcname]
	else:
		print '****************************'
		print 'Fit has converged'
		print '****************************'

	if src['ts'] > 15 and np.fabs(src['param_values'][1]) > 3.0:
		model4 = {'Index' : 2.8, 'SpatialModel' : 'PointSource'}
		print '****************************'
		print 'Source still bad TS=%s Index=%s?' %(src['ts'],src['param_values'][1])
		print '****************************'
		gta.delete_source(srcname)
		mapp=gta.tsmap('fit_no_source_final',model=model4)
		finder4 = gta.find_sources(prefix='find4',model=model4,sqrt_ts_threshold=4.0,
                      min_separation=1.0,max_iter=10,sources_per_iter=20,
                      tsmap_fitter='tsmap')
		gta.add_source(srcname,{ 'ra' : ra, 'dec' : dec,
                	'SpectrumType' : 'PowerLaw', 'Index' : 2.0,
                	'Scale' : 1000, 'Prefactor' : 1e-11,
                	'SpatialModel' : 'PointSource' })
        	gta.free_sources(minmax_ts=[100,None],pars='norm',distance=5.0)
        	gta.free_sources(minmax_ts=[200,None],distance=7.0)
		gta.free_source(srcname)
		gta.fit()
		gta.write_roi('fit_model_3')
		p = np.load('output/fit_model_3.npy').flat[0]
 		src = p['sources'][srcname]

        TS = '%.2f' %src['ts']
        Flux = '%.2e' %src['eflux']
        Flux_err = '%.2e' %src['eflux_err']
	Flux_UL = '%.2e' %src['eflux_ul95']
        Index = '%.2f' %(np.fabs(src['param_values'][1]))
        Index_err = '%.2f' %src['param_errors'][1]
        f = open('%s_Param.txt' % srcname, 'w')
        f.write(str(srcname) + "\t" + str(dist_sep) + "\t" + str(Flux) + "\t" + str(Flux_err) + "\t" + str(Index) + "\t" + str(Index_err) + "\t" + str(Flux_UL) + "\t" + str(TS) + "\n")
        f.close()
	 
        # Calculate likelihood for null hypothesis:

        if self.JLA == False:
            iteration_list = [0]
        if self.JLA == True:
            iteration_list = [0,1,2,3]
        for j in iteration_list:

	    srcmap = 'output/srcmap_0%s.fits' %j
	    bexpmap = 'output/bexpmap_0%s.fits' %j
	    xmlfile = 'output/fit_model_3_0%s.xml' %j
	    savexml = 'output/null_likelihood_%s.xml'%j
	    savetxt = 'output/null_likelihood_%s.txt' %j

	    obs = BinnedObs(srcMaps=srcmap,expCube=self.ltcube,binnedExpMap=bexpmap,irfs='P8R3_SOURCE_V2')
	    like = BinnedAnalysis(obs,xmlfile,optimizer='Minuit') 
	                    	
	    like.deleteSource(srcname)
	    freeze=like.freeze
	    for k in range(len(like.model.params)):
	        freeze(k)
	    like.model['galdiff'].funcs['Spectrum'].getParam('Prefactor').setFree(True)
	    like.model['galdiff'].funcs['Spectrum'].getParam('Index').setFree(True)
	    like.model['isodiff'].funcs['Spectrum'].getParam('Normalization').setFree(True)
	    likeobj = pyLike.Minuit(like.logLike)
	    thisL = like.fit(verbosity=1,covar=True,optObject=likeobj) #this returns -logL
	    value = like.logLike.value() #this returns logL
	    like.logLike.writeXml(savexml)

	    print "*********"
	    print "-logL: " + str(thisL)
	    print 
	
	    f = open(savetxt,'w')
	    f.write(str(value))
	    f.close()

        if self.use_scratch == True:
            shutil.copytree(src_scratch,src_output_main)
    
        os.chdir(self.home)

        return

    def make_preprocessing_summary(self):

        # Construct empty dataframe:
        df_full = pd.DataFrame(data=None, columns=["name","dist_sep","flux","flux_err","index","index_err","flux_ul","TS"])

        # Fill dataframe:
        missing_list = []
        for each in self.sample_name_list:
	
	    srcname = each
            
            # Check for updated name:
            indir = os.path.join(self.home,"Preprocessed_Sources",srcname,"output")
            new_name_file = os.path.join(indir,"replaced_source_name.txt")
            replace_name = False
            if os.path.exists(new_name_file) == True:
                f = open(new_name_file,"r")
                this_list = eval(f.read())
                newsrcname = this_list[1]
                replace_name = True

            src_file = "%s/%s_Param.txt" % (srcname,srcname)
	    wdir = os.path.join(self.home,"Preprocessed_Sources",src_file) 
	    
            if os.path.exists(wdir) == False and replace_name == True:
                src_file = "%s/%s_Param.txt" % (srcname,newsrcname)
                wdir = os.path.join(self.home,"Preprocessed_Sources",src_file)

	    if os.path.exists(wdir) == False:
	        missing_list.append(srcname)
		print 
		print "Does not exists: " + srcname
		print 
		
	    if os.path.exists(wdir) == True:
	        this_file = wdir
		df = pd.read_csv(this_file,delim_whitespace=True,names=["name","dist_sep","flux","flux_err","index","index_err","flux_ul","TS"])
		df_full = pd.concat([df,df_full]).reset_index(drop=True)

        df_full = df_full.sort_values(by=["TS"],ascending=False).reset_index(drop=True)

        print
        print df_full
        print 

        # Write dataframe to text file:
        f = open("Preprocessed_Sources/preprocessing_summary_LLAGN.txt","w")
        f.write("\n")
        f.write("*****************\n")
        f.write("preprocessing summary:\n\n")
        f.write(df_full.to_string(index=True))
        f.write("\n\n")
        f.write("****************\n")
        f.write("Missing sources:")
        f.write(str(missing_list))
        f.close()

        df_full.to_csv("Preprocessed_Sources/preprocessing_summary_LLAGN.csv",sep ="\t",index=False)
    
        return

    #################
    # Stack Sources:

    def PL2(self,Fit,name):

        """
    
        Power law spectral model for stacking: sets parameters of PowerLaw2 spectral function.
        Inputs:
            - Fit: likelihood object.
            - name: name of source. 

        """

        Fit[name].funcs['Spectrum'].getParam('Integral').setBounds(1e-14,1e7)
        Fit[name].funcs['Spectrum'].getParam('Integral').setScale(1.0)
        Fit[name].funcs['Spectrum'].getParam('Integral').setValue(1.0)
        Fit[name].funcs['Spectrum'].getParam('Integral').setFree(False)
        Fit[name].funcs['Spectrum'].getParam('Index').setBounds(-10,0)
        Fit[name].funcs['Spectrum'].getParam('Index').setScale(1.0)
        Fit[name].funcs['Spectrum'].getParam('Index').setValue(-2.0)
        Fit[name].funcs['Spectrum'].getParam('Index').setFree(False)
        Fit[name].funcs['Spectrum'].getParam('LowerLimit').setBounds(100,20000)
        Fit[name].funcs['Spectrum'].getParam('LowerLimit').setValue(self.emin)
        Fit[name].funcs['Spectrum'].getParam('LowerLimit').setFree(False)
        Fit[name].funcs['Spectrum'].getParam('LowerLimit').setScale(1.0)
        Fit[name].funcs['Spectrum'].getParam('UpperLimit').setBounds(100,5000000)
        Fit[name].funcs['Spectrum'].getParam('UpperLimit').setValue(self.emax)
        Fit[name].funcs['Spectrum'].getParam('UpperLimit').setFree(False)
        Fit[name].funcs['Spectrum'].getParam('UpperLimit').setScale(1.0)
        
        return Fit

    def run_stacking(self,srcname,PSF,indir="default"):
        
        """

        Construct 2D TS profiles for sources.

        inputs:
            - srcname: name of source.
            - PSF: integer ranging from 0-3 indicating PSF class for JLA. 
              Note: The passed value is 0 for standard analysis. 
            - indir (optional arguement): input preprocessing directory to use for stacking. 
              Note: Defualt is preprocessing directory from main run directory.
        
        """

	# Make print statement:
	print
	print "********** Fermi Stacking Analysis **********"
	print "Running stacking..."
	print

        # Define default preprocessing directory:
        if indir == "default":
            indir = os.path.join(self.home,"Preprocessed_Sources",srcname,"output")

        # Check for updated name:
        replace_name = False
        new_name_file = os.path.join(indir,"replaced_source_name.txt")
        if os.path.exists(new_name_file) == True:
            f = open(new_name_file,"r")
            this_list = eval(f.read())
            true_name = this_list[0]
            new_name = this_list[1]
            replace_name = True

        # Define main output directory for source:
        stacking_output = os.path.join(self.home,"Stacked_Sources")
        if(os.path.isdir(stacking_output)==False):
            os.system('mkdir %s' %stacking_output)
       
        if self.JLA == False:
            this_src_dir = os.path.join(stacking_output,srcname)
            if(os.path.isdir(this_src_dir)==True):
                shutil.rmtree(this_src_dir)
            os.system('mkdir %s' %this_src_dir)
        
        if self.JLA == True:
            this_likelihood = "Likelihood_" + str(PSF)
            this_likelihood_dir = os.path.join(stacking_output,this_likelihood)
            this_src_dir = os.path.join(this_likelihood_dir,srcname)
            if(os.path.isdir(this_likelihood_dir)==False):
                os.system('mkdir %s' %this_likelihood_dir)
            if(os.path.isdir(this_src_dir)==True):
                shutil.rmtree(this_src_dir)
            os.system('mkdir %s' %this_src_dir)
        
        if self.use_scratch == False:
            os.chdir(this_src_dir)

        # Define scratch directory for source:
        if self.use_scratch == True:
            stacking_scratch = os.path.join(self.scratch,"Stacking")
            if(os.path.isdir(stacking_scratch)==False):
                os.system('mkdir %s' %stacking_scratch)
            
            if self.JLA == False:
                this_scratch_dir = os.path.join(stacking_scratch,srcname)
                if(os.path.isdir(this_scratch_dir)==True):
                    shutil.rmtree(this_scratch_dir)
                os.system('mkdir %s' %this_scratch_dir)
            
                os.chdir(this_scratch_dir)

            if self.JLA == True:
                this_likelihood = "Likelihood_" + str(PSF)
                this_likelihood_dir = os.path.join(stacking_scratch,this_likelihood)
                this_scratch_dir = os.path.join(this_likelihood_dir,srcname)
                if(os.path.isdir(this_likelihood_dir)==False):
                    os.system('mkdir %s' %this_likelihood_dir)
                if(os.path.isdir(this_scratch_dir)==True):
                    shutil.rmtree(this_scratch_dir)
                os.system('mkdir %s' %this_scratch_dir)
            
                os.chdir(this_scratch_dir)        
   
	shutil.copy2('%s/srcmap_0%s.fits' %(indir,PSF), 'srcmap_0%s.fits' %PSF)
	shutil.copy2('%s/bexpmap_0%s.fits' %(indir,PSF), 'bexpmap_0%s.fits' %PSF)
	shutil.copy2('%s/fit_model_3_0%s.xml' %(indir,PSF), 'fit_model_3_0%s.xml' %PSF)

	obs = BinnedObs(srcMaps='srcmap_0%s.fits' %PSF,expCube=self.ltcube,binnedExpMap='bexpmap_0%s.fits' %PSF,irfs='P8R3_SOURCE_V2')
	
        # Define index range for scan:
        index = -1.0*np.arange(self.index_min,self.index_max+0.1,0.1)
        index = np.around(index,decimals=1)
        index = index.tolist()

	for i in range(len(index)):

                # Fix names for sources with offset positions:
                if replace_name == True:
                    srcname = new_name 
            	
		LOG_LIKE=[]
		Fit_Qual=[]
		Conv=[]
		Flux=[]
		Index=[]

                # Define flux range for scane:
		flux=np.linspace(self.flux_min,self.flux_max,num=self.num_flux_bins,endpoint=True)
		
                for j in range(len(flux)):
				
			Index+=['%.1f' %np.fabs(index[i])]
			Flux+=['%.2e' %(10**flux[j])]
			like1 = BinnedAnalysis(obs,'fit_model_3_0%s.xml' %PSF,optimizer='DRMNFB')
			freeze=like1.freeze
			for k in range(len(like1.model.params)):
				freeze(k)
			like1.setSpectrum(srcname,'PowerLaw2')
			self.PL2(like1,srcname)
			like1.model['galdiff'].funcs['Spectrum'].getParam('Prefactor').setFree(True)
			like1.model['galdiff'].funcs['Spectrum'].getParam('Index').setFree(True)
			like1.model['isodiff'].funcs['Spectrum'].getParam('Normalization').setFree(True)
			like1.model[srcname].funcs['Spectrum'].getParam('Integral').setValue(10**flux[j])
			like1.model[srcname].funcs['Spectrum'].getParam('Index').setValue(index[i])
	            
			like1.tol = 1e-2
	                like1.syncSrcParams()
	                likeobj = pyLike.Minuit(like1.logLike)
	                like1.fit(verbosity=0,covar=True,optObject=likeobj)
	                like1.logLike.writeXml('fit_1_%s.xml' %srcname)
	                
			#perform second likelihood fit with Minuit:
			like2 = BinnedAnalysis(obs,'fit_1_%s.xml'%srcname,optimizer='MINUIT')
	            
			like2.tol = 1e-8
	                like2.syncSrcParams()
	                likeobj = pyLike.Minuit(like2.logLike)
	                like2.fit(verbosity=0,covar=True,optObject=likeobj)
			Fit_Qual+=['%d' %likeobj.getQuality()]
			Conv+=['%d' %likeobj.getRetCode()]
			LOG_LIKE+=['%.2f' %like2.logLike.value()]
			del like1, like2
		
                output = '\n'.join('\t'.join(map(str,row)) for row in zip(Flux,Index,LOG_LIKE,Fit_Qual,Conv))
	
                # Fix names for sources with offset positions:
                if replace_name == True:
                    srcname = true_name

                # Write Files:
		with open('%s_stacking_%s.txt' %(srcname,np.fabs(index[i])),'w') as f:
			f.write(output)
		f.close()

                # Copy files to main output directory:
                if self.use_scratch == True:
        	    os.system('cp %s_stacking_%s.txt %s/' %(srcname,np.fabs(index[i]),this_src_dir))
		
                # Make more room in RAM:
                del Flux,Index,LOG_LIKE,Fit_Qual,Conv
        	gc.collect() #dump unused memory to speed up calculation
	
        os.chdir(self.home)
        
	return

    ###############
    # Add Stacking:

    def combine_likelihood(self, exclusion_list, savefile):
	
        """
	Make 2D TS profiles for each source and add to get stacked profile.

	Input definitions:
            - exclusion_list: list of sources to exclude from stacked profile.
	    - savefile: Prefix of array to be saved. Do not include ".npy" at the end of the name; it's already included.
	
        """
        
	# Make print statement:
	print
	print "********** Fermi Stacking Analysis **********"
	print "Combining likelihood..."
	print

        # Make main output directories:
        adding_main_dir = os.path.join(self.home,"Add_Stacking")
        if os.path.exists(adding_main_dir) == False:
            os.system("mkdir %s" %adding_main_dir)

        array_main_dir = os.path.join(adding_main_dir,"Numpy_Arrays")
        if os.path.exists(array_main_dir) == False:  
            os.system("mkdir %s" %array_main_dir)
   
        individual_array_main_dir = os.path.join(array_main_dir,"Individual_Sources")
        if os.path.exists(individual_array_main_dir) == False:  
            os.system("mkdir %s" %individual_array_main_dir)
        
        image_main_dir = os.path.join(adding_main_dir,"Images")
        if os.path.exists(image_main_dir) == False:
            os.system("mkdir %s" %image_main_dir)

        if self.JLA == False:

	    # Define counters and lists:
	    counter = 0
	    print_list = []
	    max_TS_list = []

	    # Iterate through sources:
	    for s in range(0,len(self.sample_name_list)):
			
	        srcname = self.sample_name_list[s]

                likelihood_dir = os.path.join(self.home,"Preprocessed_Sources",srcname,"output/null_likelihood_0.txt")
	        stacking_dir = os.path.join(self.home,"Stacked_Sources",srcname)
	
	        if os.path.exists(stacking_dir) == False or os.path.exists(likelihood_dir) == False:
	            print 
		    print 'Does not exist: ' + srcname
		    print 

	        if srcname not in exclusion_list and os.path.exists(likelihood_dir) == True and os.path.exists(stacking_dir) == True:
				
	            os.chdir(stacking_dir)
	        
                    print_list.append(srcname)
		
		    array_list = []
    
                    # Define index list of scan:
                    index_list = np.arange(self.index_min,self.index_max+0.1,0.1)
                    index_list = np.around(index_list,decimals=1)
                    index_list = index_list.tolist()
                
                    # Read null likelihood:
		    f = open(likelihood_dir,'r')
		    lines = f.readlines()
		    null = float(lines[0])

		    for i in range(0,len(index_list)):
		        this_index = str(index_list[i])
		        this_file = "%s_stacking_%s.txt" %(srcname,this_index)

		        df = pd.read_csv(this_file,delim_whitespace=True,names=["flux","index","likelihood","quality","status"])

		        flux = df["flux"]
		        index = df["index"]
		        likelihood = df["likelihood"].tolist()
		        TS = 2*(df["likelihood"]-null)
		        TS = TS.tolist()
		        array_list.append(TS)

	            final_array = np.array(array_list)
	
		    this_max_TS = np.max(final_array)
		    max_TS_list.append(this_max_TS)
	
		    # Save each individual source array:
                    this_file_name = srcname + "_array"
                    source_array_file = os.path.join(individual_array_main_dir,this_file_name)            	    
                    np.save(source_array_file,final_array)

                    # Get stacked array:
		    if counter == 0:
	                summed_array = final_array
		    if counter > 0:
		        summed_array = np.add(summed_array,final_array)
		    counter += 1

            print
	    print "sources that were added in the sum:"
	    print "number of sources: " + str(len(print_list))
	    print print_list
	    print 

            # Save summed array:
	    array_file = os.path.join(array_main_dir,savefile)
	    np.save(array_file,summed_array)

        if self.JLA == True:
	
            # Define counters and lists:
            total_counter = 0
	    print_list = []
	    max_TS_list = []

	    # Iterate through sources:
	    for s in range(0,len(self.sample_name_list)):
			
	        srcname = self.sample_name_list[s]
                counter = 0

                j_counter = 0
                for j in [0,1,2,3]:
            
                    likelihood_dir = "Preprocessed_Sources/%s/output/null_likelihood_%s.txt" %(srcname,str(j))
	            likelihood_dir = os.path.join(self.home,likelihood_dir)
                    stacking_dir = "Stacked_Sources/Likelihood_%s/%s" %(str(j),srcname)
	            stacking_dir = os.path.join(self.home,stacking_dir)

	            if os.path.exists(stacking_dir) == False or os.path.exists(likelihood_dir) == False:
                        print 
		        print 'Does not exist: ' + srcname + "_%s" %str(j)
		        print 
                        j_counter += 1

	            if srcname not in exclusion_list and os.path.exists(likelihood_dir) == True and os.path.exists(stacking_dir) == True:
				
	                os.chdir(stacking_dir)
	        
	                array_list = []
    
                        # Define index list of scan:
                        index_list = np.arange(self.index_min,self.index_max+0.1,0.1)
                        index_list = np.around(index_list,decimals=1)
                        index_list = index_list.tolist()
                
                        # Read null likelihood:
		        f = open(likelihood_dir,'r')
		        lines = f.readlines()
		        null = float(lines[0])

		        for i in range(0,len(index_list)):
		            this_index = str(index_list[i])
		            this_file = "%s_stacking_%s.txt" %(srcname,this_index)

		            df = pd.read_csv(this_file,delim_whitespace=True,names=["flux","index","likelihood","quality","status"])

		            flux = df["flux"]
		            index = df["index"]
		            likelihood = df["likelihood"].tolist()
		            TS = 2*(df["likelihood"]-null)
		            TS = TS.tolist()
		            array_list.append(TS)

	                final_array = np.array(array_list)
	                if counter == 0:
                            summed_array = final_array
                        if counter > 0:
                            summed_array = np.add(summed_array,final_array)
                        counter += 1
            
                if srcname not in exclusion_list and j_counter == 0:

                    print_list.append(srcname)

	            # Save each individual source array:
                    source_array_file = os.path.join(individual_array_main_dir,srcname)            	    
                    np.save(source_array_file,summed_array)

                    # Get stacked array:
		    if total_counter == 0:
	                total_summed_array = summed_array
		    if total_counter > 0:
		        total_summed_array = np.add(total_summed_array,summed_array)
		    total_counter += 1

            print
	    print "sources that were added in the sum:"
	    print "number of sources: " + str(len(print_list))
	    print print_list
	    print 

            # Save summed array:
	    array_file = os.path.join(array_main_dir,savefile)
	    np.save(array_file,total_summed_array)

        # Return to home:
	os.chdir(self.home)
		
        return 	

    #############
    # Make plots:

    def plot_final_array(self,savefig,array,use_index="default"):

        """
	 
         Input definitions:
	 
	 savefig: Name of image file to be saved. 
	
	 array: Name of input array to plot. Must include ".npy".
	
         use_index (optional_arguement): option to calculate flux for specified index.
            Note: default is best-fit index.

        """

	# Make print statement:
	print
	print "********** Fermi Stacking Analysis **********"
	print "Plotting final_array..."
	print

	# Specify the savefigure:
	savefig = os.path.join("Add_Stacking/Images/",savefig)

        # Specify array file to be plotted: 
        array_file = os.path.join("Add_Stacking/Numpy_Arrays/",array)
        if os.path.exists(array_file) == False:
            array_file = os.path.join("Add_Stacking/Numpy_Arrays/Individual_Sources/",array)
        if os.path.exists(array_file) == False:
            print
            print "Error: array file does not exists."
            print
            sys.exit()

	# Setup figure:
	fig = plt.figure(figsize=(9,9))
	ax = plt.gca()

	# Upload summed array:
	summed_array = np.load(array_file)

	# Get min and max:
	max_value = np.amax(summed_array)
	min_value = np.amin(summed_array)

	# Corresponding sigma for 2 dof:
	num_pars = 2
	sigma = stats.norm.ppf(1.-stats.distributions.chi2.sf(max_value,num_pars)/2.)

	# Significane contours for dof=2:
	first = max_value - 2.3 # 0.68 level
	second = max_value - 4.61 # 0.90 level
	third =  max_value - 9.21 # 0.99 level

	# Find indices for max values:
	ind = np.unravel_index(np.argmax(summed_array,axis=None),summed_array.shape)
        best_index_value = ind[0]
	best_flux_value = ind[1]

	# Get best index:
        index_list = np.arange(self.index_min,self.index_max+0.1,0.1)     
        best_index = index_list[ind[0]]

	# Get best flux:
        flux_list=np.linspace(self.flux_min,self.flux_max,num=40,endpoint=True)
        flux_list = 10**flux_list 
        best_flux = flux_list[ind[1]]

        # Option to calculate flux for specified index:
        if use_index != "default":
            index_list = np.around(index_list,decimals=1)
            best_index_value = np.where(index_list==use_index)[0][0]
            ind = np.unravel_index(np.argmax(summed_array[best_index_value],axis=None),summed_array.shape)
            best_flux_value = ind[1]
            best_index = index_list[best_index_value]
            best_flux = flux_list[best_flux_value]
            ind = (best_index_value,best_flux_value)
            max_value = np.amax(summed_array[best_index_value])
            sigma = stats.norm.ppf(1.-stats.distributions.chi2.sf(max_value,num_pars)/2.)

	# Smooth array:
	gauss_kernel = Gaussian2DKernel(1.5)
	filtered_arr = convolve(summed_array, gauss_kernel, boundary='extend')

	# Below I define 3 different methods to plot the array, just with different styles.
	# Use method 1 as the default.

	# Method 1
	def plot_method_1():

		img = ax.pcolormesh(flux_list,index_list,summed_array,cmap="inferno",vmin=0,vmax=max_value)
		plt.contour(flux_list,index_list,summed_array,levels = (third,second,first),colors='black',linestyles=["-.",'--',"-"], alpha=1,linewidth=2*4.0)
		plt.plot(best_flux,best_index,marker="+",ms=12,color="black")
		ax.set_xscale('log')
		plt.xticks(fontsize=16)
		plt.yticks(fontsize=16)

		return img

	# Method 2 
	def plot_method_2():

		#clip the array at zero for visualization purposes:
		for i in range(0,summed_array.shape[0]):
			for j in range(0,summed_array.shape[1]):
				if summed_array[i,j] < 0:
					summed_array[i,j] = 0

		img = ax.contourf(flux_list,index_list,summed_array,100,cmap="inferno")
		plt.contour(flux_list,index_list,summed_array,levels = (third,second,first),colors='black',linestyles=["-.",'--',"-"], alpha=1,linewidth=4.0)
		plt.plot(best_flux,best_index,marker="+",ms=12,color="black")
		plt.yticks(fontsize=14)
		return img

	# Method 3
	def plot_method_3():

		img = ax.imshow(summed_array,origin="upper",cmap='inferno',vmin=0,vmax=max_value)
		ax.contour(summed_array,levels = (third,second,first),colors='black',linestyles=["-.",'--',"-"], alpha=1,linewidth=4.0)

		return img

	# Make plot with this method:
	img = plot_method_1()

	# Plot colorbar
	cbar = plt.colorbar(img,fraction=0.045)
	cbar.set_label("TS",size=16,labelpad=12)
	cbar.ax.tick_params(labelsize=12)

	plt.ylabel('Photon Index',fontsize=22)
        plt.xlabel(r'$\mathregular{\gamma}$-Ray Flux [ph $\mathrm{cm^2}$  s$\mathregular{^{-1}}$]',fontsize=22) #for flux
	ax.set_aspect('auto')
	ax.tick_params(axis='both',which='major',length=9)
	ax.tick_params(axis='both',which='minor',length=5)
	
	plt.savefig(savefig,bbox_inches='tight')
	
        if self.show_plots == True:
            plt.show()
        
        plt.close()

	#################
	# Find 1 sigma error from the 2d array

	# Important note: the for loop is constrained to scan the respective list only once;
	# Otherwise it will loop through numerous times.

	# Get 1 sigma index upper error (lower direction of map):
	for j in range(0,len(index_list)-best_index_value):
	
		if math.fabs(summed_array[ind[0]+j][ind[1]] - max_value) < 2.3:
			pass 
		if math.fabs(summed_array[ind[0]+j][ind[1]] - max_value) >= 2.3:
			index_sigma_upper = math.fabs(index_list[ind[0]] - index_list[ind[0]+j])
			break
		if j == (len(index_list)-best_index_value-1):
			index_sigma_upper = 0 #in this case the index is not constrained toward bottom of map

	# Get 1 sigma index lower error (upper direction of map):
	for j in range(0,best_index_value):

        	if math.fabs(summed_array[ind[0]-j][ind[1]] - max_value) < 2.3:
        		pass
        	if math.fabs(summed_array[ind[0]-j][ind[1]] - max_value) >= 2.3:
        		index_sigma_lower = math.fabs(index_list[ind[0]] - index_list[ind[0]-j])
			break
		if j == best_index_value-1:
			index_sigma_lower = 0 #in this case the index is not constrained toward top of map
	

	# Get 1 sigma flux upper error:
	for j in range(0,len(flux_list)-best_flux_value):

        	if math.fabs(summed_array[ind[0]][ind[1]+j] - max_value) < 2.3:
        		pass
        	if math.fabs(summed_array[ind[0]][ind[1]+j] - max_value) >= 2.3:
        		flux_sigma_upper = math.fabs(flux_list[ind[1]] - flux_list[ind[1]+j])
        		break
		if j == (len(index_list)-best_index_value - 1):
			flux_sigma_upper = 0 #in this case the flux is not constrained toward right of map

	# Get 1 sigma flux lower error:
	for j in range(0,best_flux_value):

        	if math.fabs(summed_array[ind[0]][ind[1]-j] - max_value) < 2.3:
        		pass
        	if math.fabs(summed_array[ind[0]][ind[1]-j] - max_value) >= 2.3:
        		flux_sigma_lower = math.fabs(flux_list[ind[1]] - flux_list[ind[1]-j])
        		break
		if j == best_flux_value-1:
			flux_sigma_lower = 0 #in this case the index is not constrained toward left of map

	print 
	print "max TS: " + str(max_value) + "; sigma: " + str(sigma)
	print "indices for max TS: " + str(ind)
	print "Sanity check on indices: " + str(summed_array[ind])
	print "Best index: " + str(best_index) + ", Error:  +" + str(index_sigma_upper) + ", -" + str(index_sigma_lower)  
	print "Best flux: " + str(best_flux)  + ", Error: +" + str(flux_sigma_upper) + ", -" + str(flux_sigma_lower)
	print

	return

    def power_law_2(self,N,gamma,E,Emin,Emax):

        """ 
        
        dN/dE in units of ph/cm^s/s/MeV.
            Inputs:
            N: Integrated flux between Emin and Emax in ph/cm^2/s.
            gamma: Spectral index.
            E: Energy range in MeV. Should be array.
            Emin, Emax: Min and max energy, respectively, in MeV. 
        
        """
            
        return N*(gamma+1)*(E**gamma) / (Emax**(gamma+1) - Emin**(gamma+1))
    

    def make_butterfly(self,name):
       
        """ 
        
        Calculate butterfly plot.
            
        Inputs:
            name: name of input array (not including .npy).
                Note: this name is also used for output files. 
            numbins: number of energy bins to use for butterfly plot.
        
        """
	
        # Make print statement:
	print
	print "********** Fermi Stacking Analysis **********"
	print "Making butterfly plot..."
	print
       
        conv = 1.60218e-6 # MeV to erg
       
        # Make output directories and file:
        image_output = "Add_Stacking/Images/" + name + "_butterfly.pdf"
        
        output_data_dir = "Add_Stacking/Output_Data"
        if os.path.exists(output_data_dir) == False:
            os.system("mkdir %s" %output_data_dir)
        data_output = "Add_Stacking/Output_Data/" + name + "_butterfly.dat"

        # Define energy range and binning for butterfly plot:
        E_range = np.logspace(np.log10(self.emin),np.log10(self.emax),30) 

        # Define flux and index.
        # Must be the same that was used to make the stacked array. 
        flux_list = np.linspace(self.flux_min,self.flux_max,num=40,endpoint=True)
        flux_list = 10**flux_list
        index_list = np.arange(self.index_min,self.index_max+0.1,0.1)

        # Load stacked array:
        input_array = os.path.join("Add_Stacking/Numpy_Arrays/",name + ".npy")
        if os.path.exists(input_array) == False:
            input_array = os.path.join("Add_Stacking/Numpy_Arrays/Individual_Sources/",name + ".npy")
        if os.path.exists(input_array) == False:
            print
            print "Error: array file does not exists."
            print
            sys.exit()
        this_array = np.load(input_array)

        # Find indices for max values:
        ind = np.unravel_index(np.argmax(this_array,axis=None),this_array.shape)
        best_index = index_list[ind[0]]
        best_flux = flux_list[ind[1]]

        # Get max and significane contours for dof=2:
        max_value = np.amax(this_array)
        first = max_value - 2.3 #0.68 level
        second = max_value - 4.61 #0.90 level
        third =  max_value - 9.21 #0.99 level

        # Get indices within 1sigma contour:
        contour = np.where(this_array>=first)
     
        # Test Method 1:
        fig = plt.figure(figsize=(8,6))
        ax = plt.gca()
        plt.contour(flux_list,index_list,this_array,levels = (third,second,first),colors='black',linestyles=["-.",'--',"-"], alpha=1,linewidth=2*4.0)
        #this_array[contour] = 0
        img = ax.pcolormesh(flux_list,index_list,this_array,cmap="inferno",vmin=0,vmax=max_value)
        ax.set_xscale('log')
        plt.xlabel("Flux [$\mathrm{ph \ cm^{-2} \ s^{-1}}$]",fontsize=12)
        plt.ylabel("Index", fontsize=12)
        plt.show()
        plt.close()
    
        # Setup figure:
        fig = plt.figure(figsize=(8,6))
        ax = plt.gca()

        # Plot solutions within 1 sigma contour (sanity check):
        x = contour[1]
        y = contour[0]
        for i in range(0,len(x)):
            this_N = flux_list[x[i]]
            this_gamma = index_list[y[i]]*-1
            dnde = self.power_law_2(this_N,this_gamma,E_range,self.emin,self.emax)
            plt.loglog(E_range,conv*E_range**2 * dnde,color="red")
        
        # Interpolate array for plotting:
        x = flux_list
        y = index_list
        z = this_array
        f = interpolate.interp2d(x, y, z, kind='linear')

        # Use finer binning to fill out butterfly plot:
        plot_flux_list = np.linspace(self.flux_min,self.flux_max,num=200,endpoint=True)
        plot_flux_list = 10**plot_flux_list
        plot_index_list = np.arange(self.index_min,self.index_max+0.1,0.003)

        # Plot best-fit:
        this_N = best_flux
        this_gamma = best_index*-1
        dnde = self.power_law_2(this_N,this_gamma,E_range,self.emin,self.emax)
        plt.loglog(E_range,conv*E_range**2 * dnde,color="black",lw=2,alpha=0.7,zorder=1)
        best_flux = conv*E_range**2 * dnde

        # Plot butterfly:
        plot_list = []
        for each in plot_flux_list:
            for every in plot_index_list:
                
                this_flux = each
                this_index = every
                this_TS = f(this_flux,this_index)
                
                if this_TS >= first:
                    this_N = each
                    this_gamma = every*-1
                    dnde = self.power_law_2(this_N,this_gamma,E_range,self.emin,self.emax)
                    plt.loglog(E_range,conv*E_range**2 * dnde,color="grey",alpha=0.3,zorder=0)
                    plot_list.append(conv*E_range**2 * dnde)

        plt.ylabel(r'$\mathrm{E^2 dN/dE \ [erg \ cm^{-2} \ s^{-1}]}$',fontsize=12)
        plt.xlabel('Energy [MeV]',fontsize=12) #for flux
        ax.tick_params(axis='both',which='major',length=9)
        ax.tick_params(axis='both',which='minor',length=5)
        plt.xticks(fontsize=12)                        
        plt.yticks(fontsize=12)
        plt.ylim(5e-16,5e-11)
        plt.savefig(image_output,bbox_inches='tight')
        plt.show()
        plt.close()

        # Write butterfly to data using max and min values: 
        plot_list = np.array(plot_list)
        plot_list = plot_list.T
        min_list = []
        max_list = []
        for each in plot_list:
            this_min = min(each)
            this_max = max(each)
            min_list.append(this_min)
            max_list.append(this_max)
        
        # Check if source is too bright to make a butterfly plot:
        if len(plot_list) == 0:
            print 
            print "Warning: butterfly plot is empty! Setting to best-fit."
            print "This typically implies that the source is very bright, with very small error contours."
            print 

            min_list = best_flux
            max_list = best_flux
        
        # Write data file:
        d = {"Energy[MeV]":E_range,"Flux[erg/cm^2/s]":best_flux,"Flux_min[erg/cm^2/s]":min_list,"Flux_max[erg/cm^2/s]":max_list}
        df = pd.DataFrame(data=d,columns = ["Energy[MeV]","Flux[erg/cm^2/s]","Flux_min[erg/cm^2/s]","Flux_max[erg/cm^2/s]"])
        df.to_csv(data_output,float_format='%10.5e', sep="\t",index=False)

        return

    def get_stack_UL95(self, array_file, ul_index=2.0):

        '''
	
        Calculate one-sided 95% UL from the 2D TS arrays: 2(logL_max - logL) = 2.71.
        Note: Since the TS array is used, the factor of 2 is already included in the calculation!
        This methed is not applicable if TS<1.
	The default index is set to -2.0.

	Inputs definitions:

	array_file: 2D array to calculate UL from

        ul_index (optional arguement): spectral index to use for UL calculation.
            Note: Default value is 2.0.
	
        '''

	# Make print statement:
	print
	print "********** Fermi Stacking Analysis **********"
	print "Running get_UL..."
	print

        this_array = os.path.join("Add_Stacking/Numpy_Arrays/",array_file)
        if os.path.exists(this_array) == False:
            this_array = os.path.join("Add_Stacking/Numpy_Arrays/Individual_Sources/",array_file)
        if os.path.exists(this_array) == False:
            print
            print "Error: array file does not exists."
            print
            sys.exit()
    
        summed_array = np.load(this_array)

        # Define flux list:
        flux_list=np.linspace(self.flux_min,self.flux_max,num=40,endpoint=True)
        flux_list = 10**flux_list 

        # Get index arguement for UL calculation:
	index_list = np.arange(self.index_min,self.index_max+0.1,0.1)
        index_list = np.around(index_list,decimals=1)
        ul_arg = np.where(index_list==ul_index)[0][0]
        print
        print "Calculating UL for spectral index = " + str(ul_index)
        print

        # Extract row from 2d summed array corresponding to the UL index:
        profile = summed_array[ul_arg]
	profile =  (np.max(profile)- profile)
	flux_interp = interpolate.interp1d(flux_list,profile,bounds_error=False,kind="linear")

        # Plot profile:
	fig = plt.figure(figsize=(8,6.2))
	ax = plt.gca()

	plt.semilogx(flux_list,profile,ls="",marker="s",label="95% UL")
	plt.semilogx(flux_list,flux_interp(flux_list),ls="--",color="green") #,label="interpolation")

	plt.hlines(2.71,1e-13,1e-9,linestyles="dashdot")
	plt.grid(True,ls=":")
	plt.xlabel("$\gamma$-ray Flux [$\mathrm{ph \ cm^{-2} \ s^{-1}}$]",fontsize=16)
	plt.ylabel("2($\mathrm{logL_{max} - logL}$)",fontsize=16)
	plt.title("Profile Likelihood",fontsize=18)
	plt.legend(loc=2,frameon=False)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	ax.tick_params(axis='both',which='major',length=7)
	ax.tick_params(axis='both',which='minor',length=4)
	plt.ylim(0,15)
	plt.savefig("Add_Stacking/Images/likelihood_profile.png")
	plt.show()
	plt.close()

	# Find min of function and corresponding x at min:
	# Note: the last entry is the staring point for x, and it needs to be close to the min to converge. 
	print
	print "****************"
	print
	min_x = optimize.fmin(lambda x: flux_interp(x),np.array([1e-10]),disp=True)
    
	upper = 7e-10
	error_right = optimize.brenth(lambda x: flux_interp(x)-2.71,min_x[0],upper,xtol=1e-13) # right
	print
	print "Results:"
	print "best flux: " + str(min_x[0]) + " ph/cm^2/s"
	print "Error right: " + str(error_right - min_x[0]) + " ph/cm^2/s"
	print "95% Flux UL: " + str(error_right) + " ph/cm^2/s"
	print 	
	
	return

    def calc_upper_limit(self,srcname,ul_emin,ul_emax,comp_list=[0,1,2,3],mult_lt=False):
        	
        '''
        
	 Calculate upper limits using both a bayesian approach and a frequentist approach using results from preprocessing.
	
	 The frequentist appraoch uses the profile likelihood method, with 2.71/2 for 95% UL. This is standard in LAT analysis. 
	 However, when there is a physical boundary on a parameter (such as a normalization) the profile likelihood is always restricted 
	 to the physical region such that for a negative MLE the maximum is evaluated at zero.
	 	
	 For low significant sources the bayesian approach may be a better estimate (Thanks to Jean Ballet for pointing this out).

         Inputs:

         srcname: Name of source for UL calculation.

         ul_emin: Lower energy bound for UL calculation.

         ul_emax: Upper energy bound for UL calculation.

         comp_list (optional): List of components to add to the SummedLikelihood object for the JLA. 
            Note: Default is for typical JLA with 4 components. 
            Note: The added components must be defined in the energy range of the UL calculation.
            Note: The function supports up to 10 components (0-9). For more, further definitions must be added.

        mult_lt (optional): If using lt cubes for each component, set to True. Default is False, for single lt cube. 

	'''      

	# Make print statement:
	print
	print "********** Fermi Stacking Analysis **********"
	print "calculating upper limit..."
	print

        # Check that the included components are ok:
        if self.JLA == True:
            for each in comp_list:
                if each > 9:
                    print "***ERROR***"
                    print "Looks like you have a complex analysis!"
                    print "More than 9 components is not currently supported."
                    print "You'll need to add more definitions to the source code."
                    print
                    sys.exit()
            print
            print "Components included in the UL calculation: " + str(comp_list)
            print 

        # Check for updated name:
        preprocess_dir = os.path.join(self.home,"Preprocessed_Sources",srcname,"output")
        replace_name = False
        new_name_file = os.path.join(preprocess_dir,"replaced_source_name.txt")
        if os.path.exists(new_name_file) == True:
            f = open(new_name_file,"r")
            this_list = eval(f.read())
            true_name = this_list[0]
            new_name = this_list[1]
            replace_name = True
        
        # Define the lt cube:
        # Note: this will be overwritten for the JLA if mult_lt is set to True.
	my_expCube = self.ltcube
	
        # Analysis selections (fixed for now):
        irfs = "P8R3_SOURCE_V2"
	optimizer = "Minuit" # Minuit or NewMinuit
        conf = BinnedConfig(edisp_bins=-1) #need to account for energy dispersion!

        if self.JLA == True:

	    # Make the summedlikelihood object:
	    summed_like = SummedLikelihood()
            
            if 0 in comp_list:
                if mult_lt == True:
                    my_expCube = "Preprocessed_Sources/%s/output/ltcube_00.fits" %srcname
	        my_ExpMap_0 = "Preprocessed_Sources/%s/output/bexpmap_roi_00.fits" %srcname
    	        my_src_0 = "Preprocessed_Sources/%s/output/srcmap_00.fits" %srcname 
	        my_xml_0 = "Preprocessed_Sources/%s/output/fit_model_3_00.xml" %srcname
	        obs_0 = BinnedObs(srcMaps=my_src_0, expCube=my_expCube, binnedExpMap=my_ExpMap_0,irfs=irfs)
	        like0 = BinnedAnalysis(obs_0, my_xml_0, optimizer=optimizer, config=conf)
                like0.setEnergyRange(ul_emin,ul_emax)
                summed_like.addComponent(like0)
            
            if 1 in comp_list:
                if mult_lt == True:
                    my_expCube = "Preprocessed_Sources/%s/output/ltcube_01.fits" %srcname
	        my_ExpMap_1 = "Preprocessed_Sources/%s/output/bexpmap_roi_01.fits" %srcname
	        my_src_1 = "Preprocessed_Sources/%s/output/srcmap_01.fits" %srcname
	        my_xml_1 = "Preprocessed_Sources/%s/output/fit_model_3_01.xml" %srcname
	        obs_1 = BinnedObs(srcMaps=my_src_1, expCube=my_expCube, binnedExpMap=my_ExpMap_1,irfs=irfs)
	        like1 = BinnedAnalysis(obs_1, my_xml_1, optimizer=optimizer, config=conf)
                like1.setEnergyRange(ul_emin,ul_emax)
                summed_like.addComponent(like1)

            if 2 in comp_list:
                if mult_lt == True:
                    my_expCube = "Preprocessed_Sources/%s/output/ltcube_02.fits" %srcname
	        my_ExpMap_2 = "Preprocessed_Sources/%s/output/bexpmap_roi_02.fits" %srcname
	        my_src_2 = "Preprocessed_Sources/%s/output/srcmap_02.fits" %srcname 
	        my_xml_2 = "Preprocessed_Sources/%s/output/fit_model_3_02.xml" %srcname
	        obs_2 = BinnedObs(srcMaps=my_src_2, expCube=my_expCube, binnedExpMap=my_ExpMap_2,irfs=irfs)
	        like2 = BinnedAnalysis(obs_2, my_xml_2, optimizer=optimizer, config=conf)
                like2.setEnergyRange(ul_emin,ul_emax)
                summed_like.addComponent(like2)

            if 3 in comp_list:
                if mult_lt == True:
                    my_expCube = "Preprocessed_Sources/%s/output/ltcube_03.fits" %srcname
	        my_ExpMap_3 = "Preprocessed_Sources/%s/output/bexpmap_roi_03.fits" %srcname
	        my_src_3 = "Preprocessed_Sources/%s/output/srcmap_03.fits"   %srcname
	        my_xml_3 = "Preprocessed_Sources/%s/output/fit_model_3_03.xml" %srcname
	        obs_3 = BinnedObs(srcMaps=my_src_3, expCube=my_expCube, binnedExpMap=my_ExpMap_3,irfs=irfs)
	        like3 = BinnedAnalysis(obs_3, my_xml_3, optimizer=optimizer, config=conf)
                like3.setEnergyRange(ul_emin,ul_emax)
                summed_like.addComponent(like3)

            if 4 in comp_list:
                if mult_lt == True:
                    my_expCube = "Preprocessed_Sources/%s/output/ltcube_04.fits" %srcname
	        my_ExpMap_4 = "Preprocessed_Sources/%s/output/bexpmap_roi_04.fits" %srcname
    	        my_src_4 = "Preprocessed_Sources/%s/output/srcmap_04.fits" %srcname 
	        my_xml_4 = "Preprocessed_Sources/%s/output/fit_model_3_04.xml" %srcname
	        obs_4 = BinnedObs(srcMaps=my_src_4, expCube=my_expCube, binnedExpMap=my_ExpMap_4,irfs=irfs)
	        like4 = BinnedAnalysis(obs_4, my_xml_4, optimizer=optimizer, config=conf)
                like4.setEnergyRange(ul_emin,ul_emax)
                summed_like.addComponent(like4)

            if 5 in comp_list:
                if mult_lt == True:
                    my_expCube = "Preprocessed_Sources/%s/output/ltcube_05.fits" %srcname
	        my_ExpMap_5 = "Preprocessed_Sources/%s/output/bexpmap_roi_05.fits" %srcname
	        my_src_5 = "Preprocessed_Sources/%s/output/srcmap_05.fits" %srcname
	        my_xml_5 = "Preprocessed_Sources/%s/output/fit_model_3_05.xml" %srcname
	        obs_5 = BinnedObs(srcMaps=my_src_5, expCube=my_expCube, binnedExpMap=my_ExpMap_5,irfs=irfs)
	        like5 = BinnedAnalysis(obs_5, my_xml_5, optimizer=optimizer, config=conf)
                like5.setEnergyRange(ul_emin,ul_emax)
                summed_like.addComponent(like5)
    
            if 6 in comp_list:
                if mult_lt == True:
                    my_expCube = "Preprocessed_Sources/%s/output/ltcube_06.fits" %srcname
	        my_ExpMap_6 = "Preprocessed_Sources/%s/output/bexpmap_roi_06.fits" %srcname
	        my_src_6 = "Preprocessed_Sources/%s/output/srcmap_06.fits" %srcname 
	        my_xml_6 = "Preprocessed_Sources/%s/output/fit_model_3_06.xml" %srcname
	        obs_6 = BinnedObs(srcMaps=my_src_6, expCube=my_expCube, binnedExpMap=my_ExpMap_6,irfs=irfs)
	        like6 = BinnedAnalysis(obs_6, my_xml_6, optimizer=optimizer, config=conf)
                like6.setEnergyRange(ul_emin,ul_emax)
                summed_like.addComponent(like6)

            if 7 in comp_list:
                if mult_lt == True:
                    my_expCube = "Preprocessed_Sources/%s/output/ltcube_07.fits" %srcname
	        my_ExpMap_7 = "Preprocessed_Sources/%s/output/bexpmap_roi_07.fits" %srcname
	        my_src_7 = "Preprocessed_Sources/%s/output/srcmap_07.fits"   %srcname
	        my_xml_7 = "Preprocessed_Sources/%s/output/fit_model_3_07.xml" %srcname
	        obs_7 = BinnedObs(srcMaps=my_src_7, expCube=my_expCube, binnedExpMap=my_ExpMap_7,irfs=irfs)
	        like7 = BinnedAnalysis(obs_7, my_xml_7, optimizer=optimizer, config=conf)
                like7.setEnergyRange(ul_emin,ul_emax)
                summed_like.addComponent(like7)

            if 8 in comp_list:
                if mult_lt == True:
                    my_expCube = "Preprocessed_Sources/%s/output/ltcube_08.fits" %srcname
	        my_ExpMap_8 = "Preprocessed_Sources/%s/output/bexpmap_roi_08.fits" %srcname
	        my_src_8 = "Preprocessed_Sources/%s/output/srcmap_08.fits" %srcname 
	        my_xml_8 = "Preprocessed_Sources/%s/output/fit_model_3_08.xml" %srcname
	        obs_8 = BinnedObs(srcMaps=my_src_8, expCube=my_expCube, binnedExpMap=my_ExpMap_8,irfs=irfs)
	        like8 = BinnedAnalysis(obs_8, my_xml_8, optimizer=optimizer, config=conf)
                like8.setEnergyRange(ul_emin,ul_emax)
                summed_like.addComponent(like8)

            if 9 in comp_list:
                if mult_lt == True:
                    my_expCube = "Preprocessed_Sources/%s/output/ltcube_09.fits" %srcname
	        my_ExpMap_9 = "Preprocessed_Sources/%s/output/bexpmap_roi_09.fits" %srcname
	        my_src_9 = "Preprocessed_Sources/%s/output/srcmap_09.fits"   %srcname
	        my_xml_9 = "Preprocessed_Sources/%s/output/fit_model_3_09.xml" %srcname
	        obs_9 = BinnedObs(srcMaps=my_src_9, expCube=my_expCube, binnedExpMap=my_ExpMap_9,irfs=irfs)
	        like9 = BinnedAnalysis(obs_9, my_xml_9, optimizer=optimizer, config=conf)
                like9.setEnergyRange(ul_emin,ul_emax)
                summed_like.addComponent(like9)

	    summedobj=pyLike.Minuit(summed_like.logLike)

            # Update name:
            if replace_name == True:
                srcname = new_name

            # Fix parameters: 
            freeze=summed_like.freeze
            for k in range(len(summed_like.model.params)):
                freeze(k)

	    # Set index=2.0 for UL calculation:
	    summed_like.model[srcname].funcs['Spectrum'].getParam('Index').setValue(2.0)
	    summed_like.model[srcname].funcs['Spectrum'].getParam('Index').setFree(False)
         
            # Get TS of source:
            src_TS = summed_like.Ts(srcname)
        
	    # Calculate 95% ULs using frequentist approach:
	    ul = UpperLimits(summed_like)	
	    ul[srcname].compute(emin=ul_emin,emax=ul_emax)	

	    # Calculate 95% bayesian UL:
	    bays_ul,results = calc_int(summed_like,srcname,emin=ul_emin, emax=ul_emax,cl = 0.95)
            
        if self.JLA == False:
	    
            my_ExpMap_0 = "Preprocessed_Sources/%s/output/bexpmap_roi_00.fits" %srcname
    	    my_src_0 = "Preprocessed_Sources/%s/output/srcmap_00.fits" %srcname 
	    my_xml_0 = "Preprocessed_Sources/%s/output/fit_model_3_00.xml" %srcname
	    obs_0 = BinnedObs(srcMaps=my_src_0, expCube=my_expCube, binnedExpMap=my_ExpMap_0,irfs=irfs)
	    like0 = BinnedAnalysis(obs_0, my_xml_0, optimizer=optimizer, config=conf)
            like0.setEnergyRange(ul_emin,ul_emax)

	    likeobj=pyLike.Minuit(like0.logLike)

            # Update name:
            if replace_name == True:
                srcname = new_name

            # Fix parameters: 
            freeze=like0.freeze
            for k in range(len(like0.model.params)):
                freeze(k)
            
	    # Set index=2.0 for UL calculation:
	    like0.model[srcname].funcs['Spectrum'].getParam('Index').setValue(2.0)
	    like0.model[srcname].funcs['Spectrum'].getParam('Index').setFree(False)

	    # Calculate ULs using frequentist approach:
	    ul = UpperLimits(like0)	
	    ul[srcname].compute(emin=ul_emin,emax=ul_emax)	

	    # Calculate bayesian UL:
	    bays_ul,results = calc_int(like0,srcname,emin=ul_emin, emax=ul_emax,cl = 0.95)
    
	# Convert ul to float:
	this_string = str(ul[srcname].results[0])
	this_string = this_string.split()	
	freq_ul = float(this_string[0])

	print
	print "##########"
	print srcname
	print
        print "Source TS: " + str(src_TS)
        print
        print "Frequentist 95% UL"
	print ul[srcname].results
        print
	print "Bayesian 95% ULs:"
	print str(bays_ul) + " ph/cm^2/s" 
	print 

	return freq_ul, bays_ul 


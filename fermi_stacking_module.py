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
import matplotlib as matplotlib
matplotlib.use('agg')
from fermipy.gtanalysis import GTAnalysis
import gc
import pyLikelihood 
from BinnedAnalysis import *
from astropy.io import fits
import pandas as pd
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
       
        # Need to specify coordinates as type float
        ra = float(ra)
        dec = float(dec)

        # Make print statement:
        print
        print "Running preprocessing..."
        print

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
		yml.write("  use_local_index : True")
	        
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
                f = open("replaced_source_name.txt","w")
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
	
        if self.use_scratch == True:
            shutil.copytree(src_scratch,src_output_main)
       
        # Calculate likelihood for null hypothesis:

        if self.JLA == False:
            iteration_list = [0]
        if self.JLA == True:
            iteration_list = [0,1,2,3]
        for j in interation_list:

	    srcmap = 'srcmap_0%s.fits' %j
	    bexpmap = 'bexpmap_0%s.fits' %j
	    xmlfile = 'fit_model_3_0%s.xml' %j
	    savexml = 'null_likelihood_%s.xml'%j
	    savetxt = 'null_likelihood_%s.txt' %j

	    obs = BinnedObs(srcMaps=srcmap,expCube=ltcube,binnedExpMap=bexpmap,irfs='P8R3_SOURCE_V2')
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
        Fit[name].funcs['Spectrum'].getParam('UpperLimit').setBounds(20000,5000000)
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
		flux=np.linspace(self.flux_min,self.flux_max,num=40,endpoint=False)
		
                for j in range(len(flux)):
				
			Index+=['%.1f' %np.fabs(index[i])]
			Flux+=['%.2e' %(10**flux[j])]
			like1 = BinnedAnalysis(obs,'fit_model_3_00.xml',optimizer='DRMNFB')
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
	print "*****************"
	print "Making stack..."
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

                likelihood_dir = os.path.join("Preprocessed_Sources",srcname,"output/null_likelihood_0.txt")
	        stacking_dir = os.path.join("Stacked_Sources",srcname)
	
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
                    index_list = np.around(index,decimals=1)
                    index_list = index.tolist()
                
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
            
                    likelihood_dir = "Preprocessed_Sources/%s/output/null_likelihood_%.txt" %(srcname,str(j))
	            stacking_dir = "Stacked_Sources/Likelihood_%s/%s" %(str(j),srcname)
	
	            if os.path.exists(stacking_dir) == False or os.path.exists(likelihood_dir) == False:
	                print 
		        print 'Does not exist: ' + srcname
		        print 
                        j_counter += 1

	            if srcname not in exclusion_list and os.path.exists(likelihood_dir) == True and os.path.exists(stacking_dir) == True:
				
	                os.chdir(stacking_dir)
	        
	                array_list = []
    
                        # Define index list of scan:
                        index_list = np.arange(self.index_min,self.index_max+0.1,0.1)
                        index_list = np.around(index,decimals=1)
                        index_list = index.tolist()
                
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
                    this_file_name = srcname + "_array"
                    source_array_file = os.path.join(individual_array_main_dir,this_file_name)            	    
                    np.save(source_array_file,final_array)

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

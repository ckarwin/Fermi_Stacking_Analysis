############################################################
# 
# Written by Chris karwin; April 2022; Clemson University.
#
# Based on original code from Marco Ajello, Vaidehi Paliya, and Abhishek Desai.
#
# Purpose: Main script for Fermi-LAT stacking analysis.
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
from matplotlib.cm import register_cmap
from scipy import stats, interpolate, optimize
from matplotlib import ticker, cm
from scipy.stats.contingency import margins
import fermi_stacking.pop_studies.PopStudies as PS
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
            inputs = yaml.load(file,Loader=yaml.FullLoader)

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
        self.irfs = inputs["irfs"]
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

        if self.file_type == "tab":
            df = pd.read_csv(self.sample_file, delim_whitespace=True)
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
        print()
        print("Running preprocessing...")
        print()
 
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
        # Make src output directory:
        os.system('mkdir %s' %src_output_main)

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
            if self.ltcube != "None":
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
            yml.write("  irfs : '%s'\n" %self.irfs)
            yml.write("#--------#\n")
            yml.write("model:\n")
            yml.write("  src_radius : 15\n")
            yml.write("  src_roiwidth : 15\n")
            yml.write("  galdiff : '%s'\n" %self.galdiff)
            yml.write("  isodiff : '%s'\n" %self.isodiff)
            yml.write("  catalogs :\n")
            yml.write("    - '4FGL-DR3'\n")
            yml.write("  extdir : '/zfs/astrohe/Software/fermipy_source/lib/python3.9/site-packages/fermipy-1.1.3+2.g21485-py3.9.egg/fermipy/data/catalogs/Extended_12years/'\n")
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
                yml.write("  - { model: {isodiff: iso_P8R3_SOURCE_V3_PSF0_v1.txt},\n")
                yml.write("      selection : { evtype : 4 } }\n")
                yml.write("  - { model: {isodiff: iso_P8R3_SOURCE_V3_PSF1_v1.txt},\n")
                yml.write("      selection : { evtype : 8 } }\n")
                yml.write("  - { model: {isodiff: iso_P8R3_SOURCE_V3_PSF2_v1.txt},\n")
                yml.write("      selection : { evtype : 16 } }\n")
                yml.write("  - { model: {isodiff: iso_P8R3_SOURCE_V3_PSF3_v1.txt},\n")
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
                    print(names2[i],sepp2,r95_ns2[i])
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

        p = np.load('output/fit_model_3.npy',allow_pickle=True).flat[0]
        src = p['sources'][srcname]
        if str(src['ts'])=='nan':	# To check non-convergence
            print('****************************')
            print('Fit has not converged')
            print('****************************')
            gta.free_sources(minmax_ts=[None,100],free=False)
            gta.free_source(srcname)
            gta.fit(tol=1e-8)
            gta.write_roi('fit_model_3')
            p = np.load('output/fit_model_3.npy',allow_pickle=True).flat[0]
            src = p['sources'][srcname]
        else:
            print('****************************')
            print('Fit has converged')
            print('****************************')

        if src['ts'] > 15 and np.fabs(src['param_values'][1]) > 3.0:
            model4 = {'Index' : 2.8, 'SpatialModel' : 'PointSource'}
            print('****************************')
            print('Source still bad TS=%s Index=%s?' %(src['ts'],src['param_values'][1]))
            print('****************************')
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
            p = np.load('output/fit_model_3.npy',allow_pickle=True).flat[0]
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

            obs = BinnedObs(srcMaps=srcmap,expCube=self.ltcube,binnedExpMap=bexpmap,irfs='%s' %self.irfs)
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

            print("*********")
            print("-logL: " + str(thisL))
            print() 
	
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
                print() 
                print("Does not exists: " + srcname)
                print() 
		
            if os.path.exists(wdir) == True:
                this_file = wdir
                df = pd.read_csv(this_file,delim_whitespace=True,names=["name","dist_sep","flux","flux_err","index","index_err","flux_ul","TS"])
                df_full = pd.concat([df,df_full]).reset_index(drop=True)

        df_full = df_full.sort_values(by=["TS"],ascending=False).reset_index(drop=True)

        print()
        print(df_full)
        print()

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

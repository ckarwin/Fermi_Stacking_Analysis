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
from fermi_stacking.preprocessing.Preprocess import StackingAnalysis
from fermi_stacking.analyze_results.AnalyzeResults import Analyze

class MakeStack(StackingAnalysis,Analyze):

    """Performs stacking."""

    def PL2(self,Fit,name):

        """Power law spectral model for stacking: sets parameters of PowerLaw2 spectral function.
        
        Parameters
        ----------
        Fit : BinnedAnalysis
            Likelihood object.
        name : str 
            Name of source. 
        
        Returns
        ------
        Fit : BinnedAnalysis.  
        """

        Fit[name].funcs['Spectrum'].getParam('Integral').setBounds(1e-50,1e100)
        Fit[name].funcs['Spectrum'].getParam('Integral').setScale(1.0)
        Fit[name].funcs['Spectrum'].getParam('Integral').setValue(1.0)
        Fit[name].funcs['Spectrum'].getParam('Integral').setFree(False)
        Fit[name].funcs['Spectrum'].getParam('Index').setBounds(-10,0)
        Fit[name].funcs['Spectrum'].getParam('Index').setScale(1.0)
        Fit[name].funcs['Spectrum'].getParam('Index').setValue(-2.0)
        Fit[name].funcs['Spectrum'].getParam('Index').setFree(False)
        Fit[name].funcs['Spectrum'].getParam('LowerLimit').setBounds(100,200000)
        Fit[name].funcs['Spectrum'].getParam('LowerLimit').setValue(self.emin)
        Fit[name].funcs['Spectrum'].getParam('LowerLimit').setFree(False)
        Fit[name].funcs['Spectrum'].getParam('LowerLimit').setScale(1.0)
        Fit[name].funcs['Spectrum'].getParam('UpperLimit').setBounds(100,5000000)
        Fit[name].funcs['Spectrum'].getParam('UpperLimit').setValue(self.emax)
        Fit[name].funcs['Spectrum'].getParam('UpperLimit').setFree(False)
        Fit[name].funcs['Spectrum'].getParam('UpperLimit').setScale(1.0)
        
        return Fit

    def run_stacking(self,srcname,PSF,indir="default"):
        
        """Construct 2D TS profiles for sources.

        Parameters
        ---------
        srcname : str 
            Name of source.
        PSF : int
            Integer ranging from 0-3 indicating PSF class for JLA. 
            The passed value is 0 for standard analysis. 
        indir : str, optional
            Input preprocessing directory to use for stacking (defualt 
            is preprocessing directory from main run directory.
        """

	# Make print statement:
        print()
        print("********** Fermi Stacking Analysis **********")
        print("Running stacking...")
        print()

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
	
        obs = BinnedObs(srcMaps='srcmap_0%s.fits' %PSF,expCube=self.ltcube,binnedExpMap='bexpmap_0%s.fits' %PSF,irfs=self.irfs)
	
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

            # Define flux range for scan:
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

    def combine_likelihood(self, exclusion_list, savefile, \
            stack_mode="flux_index", likelihood_home="default"):
	
        """Make 2D TS profiles for each source and add to get stacked profile.

	Parameters
        ----------
        exclusion_list : list
            List of sources to exclude from stacked profile.
	Savefile : str
            Prefix of array to be saved. Do not include ".npy" at the 
            end of the name; it's already included.
        stack_mode : str, optional
            Type of stacking being performed. Default is flux_index. 
            Other option is alpha_beta. 
        likelihood_home : str, optional
            Full path to run directory of preprocessing, where null
            likelihood has been calculated (default is current working directory).
        """
        
        # Make print statement:
        print()
        print("Combining likelihood...")
        print()

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

        # Define default likelihood directory:
        if likelihood_home == "default":
            likelihood_home = self.home

        if self.JLA == False:

	    # Define counters and lists:
            counter = 0
            print_list = []
            max_TS_list = []

	    # Iterate through sources:
            for s in range(0,len(self.sample_name_list)):
			
                srcname = self.sample_name_list[s]

                likelihood_dir = os.path.join(likelihood_home,"Preprocessed_Sources",srcname,"output/null_likelihood_0.txt")
                stacking_dir = os.path.join(self.home,"Stacked_Sources",srcname)
	
                if os.path.exists(stacking_dir) == False or os.path.exists(likelihood_dir) == False:
                    print() 
                    print('Does not exist: ' + srcname)
                    print()

                if srcname not in exclusion_list and os.path.exists(likelihood_dir) == True and os.path.exists(stacking_dir) == True:
				
                    os.chdir(stacking_dir)
	            
                    print_list.append(srcname)
		
                    array_list = []
    
                    # Define index list of scan:
                    if stack_mode == "flux_index":
                        index_list = np.arange(self.index_min,self.index_max+0.1,0.1)
                        index_list = np.around(index_list,decimals=1)
                        index_list = index_list.tolist()
                    if stack_mode == "alpha_beta":
                        # Index implies alpha.
                        # Flux implies beta.
                        index_list = self.alpha_beta

                    # Read null likelihood:
                    f = open(likelihood_dir,'r')
                    lines = f.readlines()
                    null = float(lines[0])

                    for i in range(0,len(index_list)):
                        this_index = str(index_list[i])
                        this_file = "%s_stacking_%s.txt" %(srcname,this_index)

                        df = pd.read_csv(this_file,sep='\s+',names=["flux","index","likelihood","quality","status"])

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

            print()
            print("sources that were added in the sum:")
            print("number of sources: " + str(len(print_list)))
            print(print_list)
            print() 

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
                    likelihood_dir = os.path.join(likelihood_home,likelihood_dir)
                    stacking_dir = "Stacked_Sources/Likelihood_%s/%s" %(str(j),srcname)
                    stacking_dir = os.path.join(self.home,stacking_dir)

                    if os.path.exists(stacking_dir) == False or os.path.exists(likelihood_dir) == False:
                        print() 
                        print('Does not exist: ' + srcname + "_%s" %str(j))
                        print() 
                        j_counter += 1

                    if srcname not in exclusion_list and os.path.exists(likelihood_dir) == True and os.path.exists(stacking_dir) == True:
				
                        os.chdir(stacking_dir)

                        array_list = []

                        # Define index list of scan:
                        if stack_mode == "flux_index":
                            index_list = np.arange(self.index_min,self.index_max+0.1,0.1)
                            index_list = np.around(index_list,decimals=1)
                            index_list = index_list.tolist()
                        if stack_mode == "alpha_beta":
                            index_list = self.alpha_range

                        # Read null likelihood:
                        f = open(likelihood_dir,'r')
                        lines = f.readlines()
                        null = float(lines[0])
                        
                        for i in range(0,len(index_list)):
                            this_index = str(index_list[i])
                            this_file = "%s_stacking_%s.txt" %(srcname,this_index)
                        
                            df = pd.read_csv(this_file,sep='\s+',names=["flux","index","likelihood","quality","status"])
                        
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

            print()
            print("sources that were added in the sum:")
            print("number of sources: " + str(len(print_list)))
            print(print_list)
            print() 

            # Save summed array:
            array_file = os.path.join(array_main_dir,savefile)
            np.save(array_file,total_summed_array)

        # Return to home:
        os.chdir(self.home)
		
        return 

    def evolution_plot(self, skip_rows, savefile, preprocess_home="default",\
            exclude_list=None, use_src_names=False, stack_mode="flux_index",\
            show_index=False, show_flux=False):

        """Plot max TS as a funtion of source.
        
        Parameters
        ----------
        skip_rows : list
            List of rows to skip when reading preprocessing summary.
        savefile : str
            Prefix of output image. 
        preprocess_home : str, optional
            Full path to run directory of preprocessing.
        exlude_list : list of str, optional
            Names of sources to exclude.
        use_src_names : bool, optional
            Option to use source names in x-axis of plot 
            (default is False).
        stack_mode : str, optional
            Stacking mode to use. Default is flux_index. The other 
            option is alpha_beta.
        show_index : bool, optional
            Show index evolution (default is False).
        show_flux : bool, optional
            Show flux evolution (default is False).
        """

	# Make print statement:
        print()
        print("Running evolution_plot...")
        print()

        # Define default preprocessing home:
        if preprocess_home == "default":
            preprocess_home = self.home

        name_file = os.path.join(preprocess_home, "Preprocessed_Sources", 
                "preprocessing_summary_" + self.run_name + ".txt")
        df = pd.read_csv(name_file,sep='\s+',skiprows=skip_rows)
        name_list = df["name"]
        ts_list = df["TS"]

        # Exclude sources:
        if exclude_list != None:
            keep_index = ~np.isin(name_list,exclude_list)
            name_list = name_list[keep_index].tolist()
            ts_list = ts_list[keep_index].tolist()

        if stack_mode == "flux_index":
            index_scan = np.arange(self.index_min,self.index_max,0.1)
            flux_scan = np.linspace(self.flux_min,self.flux_max,num=41,endpoint=True)
            flux_scan = 10**flux_scan
        
        if stack_mode == "alpha_beta":
            index_scan = self.alpha_range
            flux_scan = self.beta_range

        max_list = []
        index_list = []
        flux_list = []
        name_plot_list = []
        for s in range(0,len(name_list)):
            index = len(name_list) - s - 1

            # Load array:
            this_name = name_list[index]

            # Check that source name has not been changed:
            if this_name not in self.sample_name_list:
                print("WARNING: Name has been updated and will not be included: %s" %this_name)
                continue
    
            plot_name = this_name		
            this_file = "Add_Stacking/Numpy_Arrays/Individual_Sources/" + this_name + ".npy"
            if os.path.exists(this_file) == False:
                print("WARNING: File does not exists: %s" %this_file)
                continue 
            this_array = np.load(this_file)
            name_plot_list.append(this_name)

            # Add array
            if s == 0:
                total_array = this_array
            if s > 0:
                total_array = total_array + this_array

            # Get best-fit values:
            max_value = np.amax(total_array)
	
            ind = np.unravel_index(np.argmax(total_array,axis=None),total_array.shape)
            best_index = ind[0]
            best_flux = ind[1]

            max_list.append(max_value)
            index_list.append(index_scan[best_index])
            flux_list.append(flux_scan[best_flux])


        print() 
        print("number of stacked sources: " + str(len(max_list)))
        print()

        # Plot
        fig = plt.figure(figsize=(8,6))
        ax = plt.gca()
        
        plot_range = np.arange(0,len(name_plot_list),1)

        plt.plot(plot_range,max_list,marker='s',ls="--",ms=8,color="black",zorder=10,label="Max TS (for stack)")

        plt.grid(color="grey",ls="-",alpha=0.5)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)

        # Option to use source names for x axis:
        if use_src_names == True:
            ax.set_xticks(plot_range)
            ax.set_xticklabels(ts_list,rotation=45,fontsize=12)
            ax.set_xticklabels(name_list,rotation=45,fontsize=12)

        ax.tick_params(axis='both',which='major',length=9)
        ax.tick_params(axis='both',which='minor',length=5)
        plt.xlabel("Number of Stacked Sources (TS ranked)", fontsize=14)
        plt.ylabel("Max TS",fontsize=14,color="black")
        ax.tick_params(axis='y',labelcolor="black")
        plt.xlim(0,len(max_list)+10)

        # Plot second twin axis:
        if show_index == True:
            ax2 = ax.twinx()
            ax2.plot(plot_range,index_list,marker='^',ls="-",ms=10,color="blue",alpha=0.6,label="Spectral Index")
            ax2.tick_params(axis='y',labelcolor="blue")
            if stack_mode == "flux_index":
                plt.ylabel("Index",fontsize=16,color="blue")
            if stack_mode == "alpha_beta":
                plt.ylabel("Alpha",fontsize=16,color="blue")
            plt.yticks(fontsize=12)
        
        # Plot third twin axis:
        if show_flux == True:
            ax3 = ax.twinx()
            ax3.plot(plot_range,flux_list,marker='o',ls="-",ms=10,color="darkorange",alpha=0.6,label="Flux")
            ax3.tick_params(axis='y',labelcolor="darkorange")
            if stack_mode == "flux_index":
                plt.ylabel("flux [$\mathrm{ph \ cm^{-2} \ s^{-1}}$]",fontsize=16,color="darkorange")
            if stack_mode == "alpha_beta":
                plt.ylabel("Beta",fontsize=16,color="darkorange")
            plt.yticks(fontsize=12)
            # Offset axis if also showing index:
            if show_index == True:
                ax3.spines.right.set_position(("axes", 1.2))

        plt.tight_layout()

        plt.savefig("Add_Stacking/Images/%s.pdf" %savefile)
        plt.show()	

        return


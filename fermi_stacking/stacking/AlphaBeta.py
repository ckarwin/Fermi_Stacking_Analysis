# Imports:
from fermi_stacking.stacking.Stack import MakeStack
import fermi_stacking.pop_studies.PopStudies as PS
import sys, os
import shutil
import math
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from BinnedAnalysis import *
import gc
import pyLikelihood

class MakeAlphaBeta(MakeStack):

    def alpha_beta_data(self, index, name_list, d_list, xlum): 
        
        """Initiates data for alpha-beta stacking.

        Parameters
        ----------
        index : float
            Absolute value of spectral index.
        name_list : array or list
            List of sample names. 
        d_list : array or list
            List of distance for sample in Mpc. Must be in same order 
            as name list.
        xlum : array or list
            Numpy array with luminosity values in erg/s. Must be in same order
            as name list.
        """

        # Input data:
        self.index = index
        self.name_list = name_list
        self.d_list = d_list
        self.xlum = xlum
        self.xnorm = np.mean(self.xlum)
       
        return
   
    def check_data(self):
        
        """Checks if alpha-beta data has been loaded."""
        
        try:
            self.xlum

        except:
            print()
            print("ERROR: Must first run alpha_beta_data")
            print()
            sys.exit()
        
        return
    
    def interpolate_array_alpha_beta(self, savefile, exclusion_list=[]):

        """Interpolate flux-index array to stack in alpha-beta.
        
        Parameters
        ----------
        savefile : str
            Name of output total array file (do not include .npy extension). 
        exclusion_list : list, optional
            Names of sources to exlude in total stack. 
        
        Note
        ----
        The alpha-beta data must first be specified by running alpha_beta_data. 
        """

        # Make sure alpha-beta data has been loaded:
        self.check_data()

        # Get index:
        index_list = -1.0*np.arange(self.index_min,self.index_max+0.1,0.1)
        index_list = np.around(index_list,decimals=1)
        if self.index in index_list == False:
            print("ERROR: Index not in list.")
            sys.exit()
        index_num = np.where(index_list==-1*self.index)[0][0]

        # Define flux list:
        flux_list=np.linspace(self.flux_min,self.flux_max,num=self.num_flux_bins,endpoint=True)
        for j in range(0,len(flux_list)):
            flux_list[j] = 10**flux_list[j]

        # Initialize total array:
        self.total_array = np.zeros(shape=(len(self.beta_range),len(self.alpha_range)))
        alpha_max_list = []
        alpha_min_list = []
        counter = 0

        # Iterate through sources:
        for s in range(0,len(self.name_list)):
           
            # Distance:
            this_d = self.d_list[s] * 3.086e24 # Convert Mpc to cm

            # x arguement:
            this_x = self.xlum[s]
            this_x = math.log10(this_x/self.xnorm) 
    
            # Load array:
            this_name = self.name_list[s]
            this_file = "Add_Stacking/Numpy_Arrays/Individual_Sources/" + this_name + ".npy"
            if os.path.exists(this_file) == False:
                print("WARNING: missing array for %s" %this_name)
                continue
            this_array = np.load(this_file)

            # Convert array:
            new_array = np.zeros(shape=(len(self.beta_range),len(self.alpha_range)))
            for y in range(0,len(self.beta_range)): 
                
                this_beta = self.beta_range[y]
                alpha_list = []

                for x in range(0,len(flux_list)):
                    this_flux = flux_list[x]
                    ergflux = PS.GetErgFlux(this_flux,self.index,self.emin,self.emax)
                    lum = ergflux*4*math.pi*(this_d**2)
                    this_alpha = (np.log10(lum) - this_beta)/this_x
                    alpha_list.append(this_alpha)
                
                # Interpolate function:
                f = interpolate.interp1d(alpha_list,this_array[index_num],
                    kind="linear", bounds_error=False, fill_value="extrapolate")

                # Add row to to source array for given luminosity range:
                this_row = f(self.alpha_range)
                new_array[y] = this_row

            # Save individual arrays:
            np.save("Add_Stacking/Numpy_Arrays/Individual_Sources/" + this_name + "_alpha_beta",new_array)

            # Add to total array:
            if (this_name in self.sample_name_list) & (this_name not in exclusion_list):
                counter += 1
                self.total_array += new_array

        print("Number of sources in total array: %s" %str(counter))

        # Save total array:
        array_savefile = "Add_Stacking/Numpy_Arrays/" + savefile 
        np.save(array_savefile,self.total_array)
        
        return

    def run_stacking(self, srcname, PSF, indir="default"): 
        
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

        # Make sure data has been loaded:
        self.check_data()

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

        # Get alpha-beta data from srcname:
        this_src = self.name_list == srcname 
        dist = self.d_list[this_src][0] 
        xlum = self.xlum[this_src][0]
        print()
        print("src: " + str(self.name_list[this_src]))
        print("dist: " + str(dist))
        print("xlum: " + str(xlum))
        print()

        # Convert distance to cm:
        dist = dist * 3.086e24 

        for i in range(len(self.alpha_range)):

            # Fix names for sources with offset positions:
            if replace_name == True:
                srcname = new_name 
            	
            LOG_LIKE=[]
            Fit_Qual=[]
            Conv=[]
            Beta=[]
            Alpha=[]
	
            for j in range(len(self.beta_range)):

                lum = self.alpha_range[i]*math.log10(xlum/self.xnorm) + self.beta_range[j]
                erg_flux = 10**lum / (4*math.pi*(dist**2))
                this_flux = PS.GetPhFlux(erg_flux,self.index,self.emin,self.emax)

                Alpha+=['%.1f' %(self.alpha_range[i])]
                Beta+=['%.2e' %(self.beta_range[j])]
                like1 = BinnedAnalysis(obs,'fit_model_3_0%s.xml' %PSF,optimizer='DRMNFB')
                freeze=like1.freeze
                for k in range(len(like1.model.params)):
                    freeze(k)
                like1.setSpectrum(srcname,'PowerLaw2')
                self.PL2(like1,srcname)
                like1.model['galdiff'].funcs['Spectrum'].getParam('Prefactor').setFree(True)
                like1.model['galdiff'].funcs['Spectrum'].getParam('Index').setFree(True)
                like1.model['isodiff'].funcs['Spectrum'].getParam('Normalization').setFree(True)
                like1.model[srcname].funcs['Spectrum'].getParam('Integral').setValue(this_flux)
                like1.model[srcname].funcs['Spectrum'].getParam('Index').setValue(-1*self.index)
	            
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
		
            output = '\n'.join('\t'.join(map(str,row)) for row in zip(Beta,Alpha,LOG_LIKE,Fit_Qual,Conv))
	
            # Fix names for sources with offset positions:
            if replace_name == True:
                srcname = true_name

            # Write Files:
            with open('%s_stacking_%s.txt' %(srcname,np.fabs(self.alpha_range[i])),'w') as f:
                f.write(output)
            f.close()

            # Copy files to main output directory:
            if self.use_scratch == True:
                os.system('cp %s_stacking_%s.txt %s/' %(srcname,np.fabs(self.alpha_range[i]),this_src_dir))
		
            # Make more room in RAM:
            del Beta,Alpha,LOG_LIKE,Fit_Qual,Conv
            gc.collect() #dump unused memory to speed up calculation
	
        os.chdir(self.home)
        
        return

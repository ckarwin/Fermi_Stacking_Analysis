# Imports
from time import sleep
from random import randint
import resource
import random,shutil,yaml
import os,sys
from math import *
import numpy as np
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
import fermi_stacking.pop_studies as PS
from IntegralUpperLimit import calc_int
from UpperLimits import UpperLimits
from SummedLikelihood import *
from astropy.io import fits
from matplotlib.ticker import FormatStrFormatter

class Analyze():    
    
    """Analyzes stacked results."""

    def plot_final_array(self,savefig,array,use_index="default",stack_mode="flux_index"):

        """Plots the stacked profile.
	 
        Parameters
        ----------
	savefig : str
            Name of image file to be saved. 
	array : str
            Name of input array to plot. Must include ".npy".
	use_index : float, optional
            Option to calculate flux for specified index (default is 
            best-fit index).
        stack_mode : str, optional
            Mode of stacking. Default is flux_index. Other options are
            alpha_beta or alpha_beta_interp. 
        """
        
        # Make print statement:
        print()
        print("Plotting final_array...")
        print()
        
        # Specify the savefigure:
        savefig = os.path.join("Add_Stacking/Images/",savefig)
        
        # Specify array file to be plotted: 
        array_file = os.path.join("Add_Stacking/Numpy_Arrays/",array)
        if os.path.exists(array_file) == False:
            array_file = os.path.join("Add_Stacking/Numpy_Arrays/Individual_Sources/",array)
        if os.path.exists(array_file) == False:
            print()
            print("Error: array file does not exists.")
            print()
            sys.exit()
        
        # Setup figure:
        fig = plt.figure(figsize=(9,9))
        ax = plt.gca()
        
        # Upload summed array:
        summed_array = np.load(array_file)
        if stack_mode == "alpha_beta_interp":
            summed_array = summed_array.T

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
        if stack_mode == "flux_index":
            index_list = np.arange(self.index_min,self.index_max+0.1,0.1)     
        if stack_mode in ["alpha_beta","alpha_beta_interp"]:
            index_list = self.alpha_range
        best_index = index_list[ind[0]]
        
        # Get best flux:
        if stack_mode == "flux_index":
            flux_list=np.linspace(self.flux_min,self.flux_max,num=40,endpoint=True)
            flux_list = 10**flux_list 
        if stack_mode in ["alpha_beta","alpha_beta_interp"]:
            flux_list = self.beta_range
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
            plt.contour(flux_list,index_list,summed_array,levels = (third,second,first),
                    colors='limegreen',linestyles=["-.",'--',"-"], alpha=1,linewidths=2)
            plt.plot(best_flux,best_index,marker="+",ms=12,color="black")
            if stack_mode == "flux_index":
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
            plt.contour(flux_list,index_list,summed_array,levels = (third,second,first),colors='black',linestyles=["-.",'--',"-"], alpha=1,linewidths=4.0)
            plt.plot(best_flux,best_index,marker="+",ms=12,color="black")
            plt.yticks(fontsize=14)
            
            return img

	# Method 3
        def plot_method_3():

            img = ax.imshow(summed_array,origin="upper",cmap='inferno',vmin=0,vmax=max_value)
            ax.contour(summed_array,levels = (third,second,first),colors='black',linestyles=["-.",'--',"-"], alpha=1,linewidths=4.0)

            return img
        
        # Make plot with this method:
        img = plot_method_1()
        
        # Plot colorbar
        cbar = plt.colorbar(img,fraction=0.045)
        cbar.set_label("TS",size=16,labelpad=12)
        cbar.ax.tick_params(labelsize=12)
        
        if stack_mode == "flux_index":
            plt.ylabel('Photon Index',fontsize=22)
            plt.xlabel(r'$\mathregular{\gamma}$-Ray Flux [ph $\mathrm{cm^2}$  s$\mathregular{^{-1}}$]',fontsize=22)
        if stack_mode in ["alpha_beta","alpha_beta_interp"]:
            plt.ylabel(r"$\alpha$",fontsize=22)
            plt.xlabel(r"$\beta$",fontsize=22)
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
        
        try: 
            print() 
            print("max TS: " + str(max_value) + "; sigma: " + str(sigma))
            print("indices for max TS: " + str(ind))
            print("Sanity check on indices: " + str(summed_array[ind]))
            print("Best index: " + str(best_index) + ", Error:  +" + str(index_sigma_upper) + ", -" + str(index_sigma_lower))  
            print("Best flux: " + str(best_flux)  + ", Error: +" + str(flux_sigma_upper) + ", -" + str(flux_sigma_lower))
            print()
        except:
            print("WARNING: Something wrong with bounds. Double check!")
        
        return

    def power_law_2(self,N,gamma,E,Emin,Emax):

        """Function for dN/dE in units of ph/cm^s/s/MeV.
        
        Parameters
        ----------
        N : float
            Integrated flux between Emin and Emax in ph/cm^2/s.
        gamma : float
            Spectral index.
        E : array
            Energy range in MeV. 
        Emin : float
            Minimum energy in MeV. 
        Emax: Maximum energy in MeV. 
        
        Returns
        -------
        array
            Function for dN/dE.
        """
            
        return N*(gamma+1)*(E**gamma) / (Emax**(gamma+1) - Emin**(gamma+1))
    

    def make_butterfly(self,name,fig_kwargs={},show_contour=False):
       
        """Calculate butterfly plot.
            
        Parameters
        ----------
        name : str 
            name of input array (not including .npy). Note: this name is also used for output files.  
        fig_kwargs : dict, optional
            pass any kwargs to plt.gca().set()
        show_contour : bool, optional
            Sets contour region to zero, as sanity check (default is False).
        """
	
        # Make print statement:
        print()
        print("Making butterfly plot...")
        print()
        
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
        flux_list = np.linspace(self.flux_min,self.flux_max,num=self.num_flux_bins,endpoint=True)
        flux_list = 10**flux_list
        index_list = np.arange(self.index_min,self.index_max+0.1,0.1)
        
        # Load stacked array:
        input_array = os.path.join("Add_Stacking/Numpy_Arrays/",name + ".npy")
        if os.path.exists(input_array) == False:
            input_array = os.path.join("Add_Stacking/Numpy_Arrays/Individual_Sources/",name + ".npy")
        if os.path.exists(input_array) == False:
            print()
            print("Error: array file does not exists.")
            print()
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
        
        # Test Method:
        fig = plt.figure(figsize=(8,6))
        ax = plt.gca()
        plt.contour(flux_list,index_list,this_array,levels = (third,second,first),colors='black',linestyles=["-.",'--',"-"], alpha=1,linewidth=2*4.0)
        if show_contour == True:
            this_array[contour] = 0
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
        ax.set(**fig_kwargs)
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
            print() 
            print("Warning: butterfly plot is empty! Setting to best-fit.")
            print("This typically implies that the source is very bright, with very small error contours.")
            print()
        
            min_list = best_flux
            max_list = best_flux
        
        # Write data file:
        d = {"Energy[MeV]":E_range,"Flux[erg/cm^2/s]":best_flux,"Flux_min[erg/cm^2/s]":min_list,"Flux_max[erg/cm^2/s]":max_list}
        df = pd.DataFrame(data=d,columns = ["Energy[MeV]","Flux[erg/cm^2/s]","Flux_min[erg/cm^2/s]","Flux_max[erg/cm^2/s]"])
        df.to_csv(data_output,float_format='%10.5e', sep="\t",index=False)
        
        return

    def get_stack_UL95(self, array_file, ul_index=2.0):

        """Calculate one-sided 95% UL from the 2D TS arrays: 2(logL_max - logL) = 2.71.

	Parameters
        ----------
        array_file : array 
            2D array to calculate UL from.
        ul_index : float, optional
            Spectral index to use for UL calculation (default value is 2.0).
       
        Note
        ----
        Since the TS array is used, the factor of 2 is already included in the calculation!
        
        This methed is not applicable if TS<1.
       """
        
        # Make print statement:
        print()
        print("Running get_UL...")
        print()
        
        this_array = os.path.join("Add_Stacking/Numpy_Arrays/",array_file)
        if os.path.exists(this_array) == False:
            this_array = os.path.join("Add_Stacking/Numpy_Arrays/Individual_Sources/",array_file)
        if os.path.exists(this_array) == False:
            print()
            print("Error: array file does not exists.")
            print()
            sys.exit()
        
        summed_array = np.load(this_array)
        
        # Define flux list:
        flux_list=np.linspace(self.flux_min,self.flux_max,num=40,endpoint=True)
        flux_list = 10**flux_list 
        
        # Get index arguement for UL calculation:
        index_list = np.arange(self.index_min,self.index_max+0.1,0.1)
        index_list = np.around(index_list,decimals=1)
        ul_arg = np.where(index_list==ul_index)[0][0]
        print()
        print("Calculating UL for spectral index = " + str(ul_index))
        print()
        
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
        print()
        print("****************")
        print()
        min_x = optimize.fmin(lambda x: flux_interp(x),np.array([1e-10]),disp=True)
        
        upper = 7e-10
        error_right = optimize.brenth(lambda x: flux_interp(x)-2.71,min_x[0],upper,xtol=1e-13) # right
        print()
        print("Results:")
        print("best flux: " + str(min_x[0]) + " ph/cm^2/s")
        print("Error right: " + str(error_right - min_x[0]) + " ph/cm^2/s")
        print("95% Flux UL: " + str(error_right) + " ph/cm^2/s")
        print()	
        
        return
        
    def calc_upper_limit(self,srcname,ul_emin,ul_emax,comp_list=[0,1,2,3],mult_lt=False):
        	
        """Calculate upper limits using both a bayesian approach and 
        a frequentist approach using results from preprocessing.
	
	The frequentist appraoch uses the profile likelihood method, with 
        2.71/2 for 95% UL. This is standard in LAT analysis. However, 
        when there is a physical boundary on a parameter (such as 
        a normalization) the profile likelihood is always restricted 
	to the physical region such that for a negative MLE the maximum is 
        evaluated at zero.
	 	
	For low significant sources the bayesian approach may be a better 
        estimate (Thanks to Jean Ballet for pointing this out).

        Parameters
        ----------
        srcname : str
            Name of source for UL calculation.
        ul_emin : float
            Lower energy bound for UL calculation in MeV.
        ul_emax : float
            Upper energy bound for UL calculation in MeV.
        comp_list : list, optional
            List of components to add to the SummedLikelihood object 
            for the JLA (default is for typical JLA with 4 components). 
            The added components must be defined in the energy
            range of the UL calculation. The function supports up to 
            10 components (0-9). For more, further definitions must be added.
        mult_lt : bool, optional
            If using lt cubes for each component, set to True (default is False, 
            for single lt cube. 

        Note 
        ----
        If getting 'IndexError', try moving the lower energy bound slightly below the bin. 

        Returns
        -------
        float, float
            Frequentist and Bayesian upper limit value, respectively. 
        """      
        
        # Make print statement:
        print()
        print("calculating upper limit...")
        print()
        
        # Check that the included components are ok:
        if self.JLA == True:
            for each in comp_list:
                if each > 9:
                    print("***ERROR***")
                    print("Looks like you have a complex analysis!")
                    print("More than 10 components is not currently supported.")
                    print("You'll need to add more definitions to the source code.")
                    print()
                    sys.exit()
            print()
            print("Components included in the UL calculation: " + str(comp_list))
            print() 
        
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
        irfs = self.irfs
        optimizer = "NewMinuit" # Minuit or NewMinuit
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
        
        print()
        print("##########")
        print(srcname)
        print()
        print("Source TS: " + str(src_TS))
        print()
        print("Frequentist 95% UL")
        print(ul[srcname].results)
        print()
        print("Bayesian 95% ULs:")
        print(str(bays_ul) + " ph/cm^2/s") 
        print() 
        
        return freq_ul, bays_ul 

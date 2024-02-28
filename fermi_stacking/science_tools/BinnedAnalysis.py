# Imports:
import gt_apps as my_apps
from GtApp import GtApp
import xml.etree.ElementTree as ET
import pyLikelihood
from BinnedAnalysis import *
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import os
from pylab import *
import time
import matplotlib.gridspec as gridspec
from astropy.wcs import WCS
from astropy.io import ascii,fits
from astropy.table import Table
from fermi_stacking.preprocessing.Preprocess import StackingAnalysis

class MakeBinnedAnalysis(StackingAnalysis):

    def gtselect(self):
		
        """Makes main date selections."""

        my_apps.filter['evclass'] = self.evclass
        my_apps.filter['ra'] = self.ROI_RA 
        my_apps.filter['dec'] = self.ROI_DEC 
        my_apps.filter['rad'] = self.ROI_radius  
        my_apps.filter['emin'] = self.emin
        my_apps.filter['emax'] = self.emax
        my_apps.filter['zmax'] = self.zmax
        my_apps.filter['tmin'] = self.tmin
        my_apps.filter['tmax'] = self.tmax 
        my_apps.filter['infile'] = self.ft1
        my_apps.filter['evtype'] = self.evtype
        my_apps.filter['outfile'] =  '%s_binned_filtered.fits' % self.run_name
        my_apps.filter.run()
        
        return 'outfile'
    
    def maketime(self):
    
        """Selects good time intervals (GTIs) and makes data quality cuts."""
    	
        my_apps.maketime['scfile'] = self.ft2
        my_apps.maketime['roicut'] = 'no'
        my_apps.maketime['evfile'] = '%s_binned_filtered.fits' % self.run_name
        my_apps.maketime['outfile'] = '%s_binned_gti.fits' % self.run_name
        my_apps.maketime['filter'] = '(DATA_QUAL>0) && (LAT_CONFIG==1)' 
        my_apps.maketime.run()
    	
        return 'outfile'
    
    #make a 2D counts map
    
    def cmap(self):
    
        """Calculates 2D counts map."""

        my_apps.counts_map['algorithm'] = 'CMAP'
        my_apps.counts_map['evfile'] = '%s_binned_gti.fits' % self.run_name
        my_apps.counts_map['outfile'] = '%s_cmap.fits' % self.run_name
        my_apps.counts_map['nxpix'] = self.nxpix
        my_apps.counts_map['nypix'] = self.nypix
        my_apps.counts_map['binsz'] = self.binsz
        my_apps.counts_map['coordsys'] = self.coordsys
        my_apps.counts_map['xref'] = self.xref
        my_apps.counts_map['yref'] = self.yref	
        my_apps.counts_map['proj'] = self.proj
        my_apps.counts_map.run()
    	
        return 'outfile'
    	
    def ccube(self):
        
        """Make a 3D counts map.
        
        Notes
        -----
        It is very important to change the number of pixels in this step!
        """
    	
        my_apps.counts_map['algorithm'] = 'CCUBE'
        my_apps.counts_map['evfile'] = '%s_binned_gti.fits' % self.run_name
        my_apps.counts_map['outfile'] = '%s_ccube.fits' % self.run_name
        my_apps.counts_map['nxpix'] = self.reduced_x
        my_apps.counts_map['nypix'] = self.reduced_y
        my_apps.counts_map['binsz'] = self.binsz
        my_apps.counts_map['coordsys'] = self.coordsys
        my_apps.counts_map['xref'] = self.xref
        my_apps.counts_map['yref'] = self.yref
        my_apps.counts_map['proj'] = self.proj
        my_apps.counts_map['ebinalg'] = 'LOG'
        my_apps.counts_map['emin'] = self.emin
        my_apps.counts_map['emax'] = self.emax
        my_apps.counts_map['enumbins'] = self.enumbins
        my_apps.counts_map.run()
    	
        return 'outfile'
    
    def expCube(self):
        
        """Calculates livetime cube."""

        my_apps.expCube['evfile'] = '%s_binned_gti.fits' % self.run_name
        my_apps.expCube['scfile'] =  self.ft2
        my_apps.expCube['outfile'] = '%s_binned_ltcube.fits' % self.run_name
        my_apps.expCube['dcostheta'] = 0.025
        my_apps.expCube['zmax'] = self.zmax
        my_apps.expCube['binsz'] = 1
        my_apps.expCube.run()
        
        return 'outfile'    

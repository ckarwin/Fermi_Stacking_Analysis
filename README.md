# fermi-stacking

## Introduction
This repository contains code for running a gamma-ray stacking analysis with Fermi-LAT data. The technique was originally developed by Marco Ajello, Vaidehi Paliya, and Abhishek Desai, and it has been successfully applied to the following studies: <br />
* Extreme blazars [(link)](https://arxiv.org/pdf/1908.02496.pdf)  <br />
* Star-forming galaxies [(link)](https://arxiv.org/pdf/2003.05493.pdf) <br />
* Extragalactic background light [(link)](https://arxiv.org/pdf/1812.01031.pdf) <br />
* Ultra-fast outflows [(link)](https://iopscience.iop.org/article/10.3847/1538-4357/ac1bb2) <br />
* Molecular outflows [(link)](https://iopscience.iop.org/article/10.3847/1538-4357/acaf57) <br />
* FRO radio galaxies [(link)](https://arxiv.org/abs/2310.19888) <br />

## Methodology 
The main assumption made with the stacking technique is that the source population can be characterized by average quantities, such as average flux and spectral index. Of course other parameters can also be stacked. 2D TS profiles are then constructed for each source using a binned likelihood analysis, and the individual profiles are summed to obtain the global significance of the signal. See above references for more details.

## Requirements
The stacking analysis requires Fermipy, available [here](https://fermipy.readthedocs.io), and it is meant to be ran on a cluster. Specifically, the package is compatible with fermipy v1.2.2, which is based on fermitools v2.2.0, and uses python 3. <br />

## Documentation
Documentation can be found here: https://fermi-stacking-analysis.readthedocs.io/en/latest/

## Installation
Using pip 
```
pip install fermi-stacking
```
From source (for developers)
```
git clone https://github.com/ckarwin/Fermi_Stacking_Analysis.git
cd Fermi_Stacking_Analysis
pip install -e .
```

## Quickstart Guide <br /> 
<pre>
1. For any new analysis: </b>
 - Make a new analysis directory (e.g. mkdir Run_1)
 - Run command line prompt 'make_stacking_run', which will setup the directory with all needed files.

2. Specify inputs in inputs.yaml. </b>
  
3. Uncomment functions inside the client code that you want to run. </b>

4. To run the code: 
 - To submit batch jobs: batch script templates are provided for both SLURM and PBS.
 - To run from terminal: python client.py
</pre>

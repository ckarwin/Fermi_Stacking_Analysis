## Introduction
This repository contains code for running a gamma-ray stacking analysis with Fermi-LAT data. The technique was originally developed by Marco Ajello, Vaidehi Paliya, and Abhishek Desai, and it has been successfully applied to the following studies: <br />
* Extreme blazars [(link)](https://arxiv.org/pdf/1908.02496.pdf)  <br />
* Star-forming galaxies [(link)](https://arxiv.org/pdf/2003.05493.pdf) <br />
* Extragalactic background light [(link)](https://arxiv.org/pdf/1812.01031.pdf) <br />
* Ultra-fast outflows [(link)](https://iopscience.iop.org/article/10.3847/1538-4357/ac1bb2) <br />

The stacking analysis requires Fermipy, available [here](https://fermipy.readthedocs.io/en/latest/). <br />
 - The current version of the code is compatible with fermipy v0.19.0, fermitools v1.2.23, and python 2.7 or later. 
 - An update will soon be available for the latest version of fermipy and python 3. 

The stacking analysis is meant to be ran on a cluster. In particular, it has been developed using the Clemson University Palmetto Cluster. More information on the Palmetto Cluster can be found [here](https://www.palmetto.clemson.edu/palmetto/basic/started/).  <br />


## Methodology 
The main assumption made with the stacking technique is that the source population can be characterized by average quantities, such as average flux and spectral index. Of course other parameters can also be stacked. 2D TS profiles are then constructed for each source using a binned likelihood analysis, and the individual profiles are summed to obtain the global significance of the signal. See above references for more details.  

## Quickstart Guide <br /> 
<pre>
1. Download Fermi_Stacking_Analysis directory:
  - download directly or git clone https://github.com/ckarwin/Fermi_Stacking_Analysis.git

2. It's advised to add the Fermi_Stacking_Analysis directory to your python path: </b>
  - add to your .bashrc file (in home directory): export PYTHONPATH=$PYTHONPATH:full_path/Fermi_Stacking_Analysis
 
3. For any new analysis (assuming you added your path), copy the following files to a new analysis directory: </b>
 - client_code.py </b>
 - inputs.yaml </b>
 - submit_fermi_stacking_jobs.py </b>

4. Specify inputs in inputs.yaml. </b>
 - This is the only file a user should have to modify (apart from running functions in the client code, as described below).
 
5. Uncomment functions inside the client code that you want to run. </b>

6. To run the code: 
 - To submit batch jobs: python submit_fermi_stacking_jobs.py 
 - To run from terminal: python client.py
</pre>

## Development Notes
* The source code is still in development.
* Please let me know if there are any errors to be fixed and/or functionallity that should be added.
* Contributions are welcome! To do so please make a standard pull request, as described in more detail [here](https://gist.github.com/MarcDiethelm/7303312).

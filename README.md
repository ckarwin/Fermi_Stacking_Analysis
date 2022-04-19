## Introduction
This repository contains code for running a gamma-ray stacking analysis with Fermi-LAT data. The technique was originally developed by Marco Ajello, Vaidehi Paliya, and Abhishek Desai, and it has been successfully applied to the following studies: <br />
* Extreme blazars [(link)](https://arxiv.org/pdf/1908.02496.pdf)  <br />
* Star-forming galaxies [(link)](https://arxiv.org/pdf/2003.05493.pdf) <br />
* Extragalactic background light [(link)](https://arxiv.org/pdf/1812.01031.pdf) <br />
* Ultra-fast outflows [(link)](https://iopscience.iop.org/article/10.3847/1538-4357/ac1bb2) <br />

The stacking analysis requires Fermipy, available [here](https://fermipy.readthedocs.io/en/latest/). <br />

The stacking analysis is meant to be ran on a cluster. In particular, it has been developed using the Clemson University Palmetto Cluster. More information on the Palmetto Cluster can be found [here](https://www.palmetto.clemson.edu/palmetto/basic/started/).  <br />


## Methodology 
The main assumption made with the stacking technique is that the source population can be characterized by average quantities, such as average flux and spectral index. Of course other parameters can also be stacked. 2D TS profiles are then constructed for each source using a binned likelihood analysis, and the individual profiles are summed to obtain the global significance of the signal. See above references for more details.  

## Quickstart Guide <br /> 
<pre>
1. Download Fermi_Stacking_Analysis directory:
  - download directly or git clone https://github.com/ckarwin/Fermi_Stacking_Analysis.git

2. Everything needed to run the analysis is in Fermi_Stacking_Analysis </b> 
  - It's advised to add the Fermi_Stacking_Analysis directory to your python path.
  - For any new analysis (assuming you added your path), copy the following files to a new analysis directory: client_code.py, inputs.yaml, and submit_fermi_stacking_jobs.py.

3. Specify inputs in inputs.yaml. </b>

4. Specify the source sample at the top of submit_fermi_stacking_jobs.py. </b>

5. To run the code, uncomment all functions inside the client code you want to run. </b>

6. Batch jobs are submitted with submit_fermi_stacking_jobs.py, and the client code can also be ran from the terminal, i.e. to make plots, etc.
</pre>


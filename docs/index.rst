.. fermi_stacking documentation master file, created by
   sphinx-quickstart on Mon Feb  5 09:07:43 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fermi-stacking's documentation!
==========================================

Introduction
------------
This package contains code for running a gamma-ray stacking analysis with Fermi-LAT data. The technique has been successfully applied to the following studies: 

- `Extreme blazars <https://arxiv.org/pdf/1908.02496.pdf>`_
- `Star-forming galaxies <https://arxiv.org/pdf/2003.05493.pdf>`_
- `Extragalactic background <https://arxiv.org/pdf/1812.01031.pdf>`_
- `Ultra-fast outflows <https://iopscience.iop.org/article/10.3847/1538-4357/ac1bb2>`_
- `Molecular outflows <https://iopscience.iop.org/article/10.3847/1538-4357/acaf57>`_
- `FR0 Radio galaxies <https://arxiv.org/abs/2310.19888>`_
- `Dark matter searches <https://arxiv.org/abs/2311.04982>`_

Methodology
-----------
The main assumption made with the stacking technique is that the source population can be characterized by average quantities, such as average flux and spectral index. Other parameters can also be stacked. 2D test statistic (TS) profiles are then constructed for each source using a binned likelihood analysis, and the individual profiles are summed to obtain the total significance of the signal. See above references for more details.

Requirements
------------
The stacking pipeline requires Fermipy (available `here <https://fermipy.readthedocs.io>`_), and it is meant to be run on a cluster. Specifically, the package is compatible with fermipy v1.2.2, which is based on fermitools v2.2.0, and uses python 3. 

Getting Help
------------
For issues with the code please open an issue in github. For further assistance, please email Chris Karwin at christopher.m.karwin@nasa.gov. 

.. warning::
   While many features are already available, fermi-stacking is still actively under development. Note that the current releases are not stable and various components can be modified or deprecated shortly.

Contributing
------------
This library is open source and anyone can contribute. If you have code you would like to contribute, please fork the repository and open a pull request. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   quickstart/quickstart
   api/index

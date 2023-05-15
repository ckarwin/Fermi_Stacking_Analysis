# Imports:
from setuptools import setup, find_packages

# Setup:
setup(
    name='atmospheric_gammas',
    version="dev",
    url='https://github.com/ckarwin/Fermi_Stacking_Analysis.git',
    author='Chris Karwin',
    author_email='christopher.m.karwin@nasa.gov',
    packages=find_packages(),
    description = "Performs Fermi-LAT stacking analysis."
)

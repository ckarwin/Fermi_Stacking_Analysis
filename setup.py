# Imports:
from setuptools import setup, find_packages

# Setup:
setup(
    name='fermi_stacking',
    version="v0.1.5-alpha",
    url='https://github.com/ckarwin/Fermi_Stacking_Analysis.git',
    author='Chris Karwin',
    author_email='christopher.m.karwin@nasa.gov',
    packages=find_packages(),
    install_requires = ["pandas","pyyaml"],
    description = "Performs Fermi-LAT stacking analysis.",
    entry_points = {"console_scripts":["make_stacking_run = fermi_stacking.make_new_run:main"]}
)

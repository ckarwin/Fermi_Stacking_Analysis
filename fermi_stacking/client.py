# Imports:
from fermi_stacking.stacking.Stack import MakeStack
from fermi_stacking.science_tools.BinnedAnalysis import MakeBinnedAnalysis
from fermi_stacking.stacking.AlphaBeta import MakeAlphaBeta
import sys 

def main(cmd_line):

    # Code can be ran with batch system or from command line

    # For batch system:
    if len(cmd_line) == 5:
        srcname = cmd_line[1]
        ra = cmd_line[2]
        dec = cmd_line[3]
        psf = cmd_line[4]

    # Define instance with input parameter card:
    instance = MakeStack("inputs.yaml")
    
    # Define exclusion list:
    exclusion_list = []

    ###########################################
    # Uncomment below to run functions:

    # For pre-computed ltcube:
    #binned_instance = MakeBinnedAnalysis("inputs.yaml")
    #binned_instance.gtselect()
    #binned_instance.maketime()
    #binned_instance.expCube()

    # Standard Stacking routines:
    #instance.run_preprocessing(srcname,ra,dec)
    #instance.make_preprocessing_summary() 
    #instance.run_stacking(srcname,psf)
    #instance.combine_likelihood(exclusion_list,savefile)
    #instance.plot_final_array("name.png","name.npy")
    #instance.evolution_plot([0,1],exclude_list=exclusion_list)
    #instance.make_butterfly(name)
    #instance.get_stack_UL95("name.npy")
    #instance.calc_upper_limit(srcname,emin,emax)
    
    # Alpha-Beta stacking:
    #instance = MakeAlphaBeta("inputs.yaml")
    #instance.alpha_beta_data(index, name_list, d_list, lum_list)
    #instance.interpolate_array_alpha_beta("run_name")
    #instance.run_stacking(srcname,psf,indir=preproceesing_dir)
    #instance.combine_likelihood(exclusion_list,"full_sample",\
    #        stack_mode="alpha_beta",likelihood_home=likelihood_dir)
    #instance.plot_final_array("full_sample.png","full_sample.npy",\
    #        stack_mode="alpha_beta")
    ###########################################

########################
if __name__=="__main__":
        main(sys.argv)

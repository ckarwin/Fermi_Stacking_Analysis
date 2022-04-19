# Imports:
from fermi_stacking_module import StackingAnalysis
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
    instance = StackingAnalysis("inputs.yaml")

    ###########################################
    # Uncomment below to run functions:
    
    #instance.run_preprocessing(srcname,ra,dec)
    #instance.make_preprocessing_summary()
    instance.run_stacking(srcname,psf)
    #instance.add_stack()
    
    ###########################################

########################
if __name__=="__main__":
        main(sys.argv)

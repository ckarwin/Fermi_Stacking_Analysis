import os
import fermi_stacking

install_dir = os.path.split(fermi_stacking.__file__)[0]

def main():

    # Copy starting files to new analysis directory:
    new_dir = os.getcwd()
    os.system("scp %s/inputs.yaml %s/client.py %s/submit_fermi_stacking_jobs.py \
            %s/array_job.pbs %s/single_job.pbs %s" \
            %(install_dir,install_dir,install_dir,install_dir,install_dir,new_dir))

########################
if __name__=="__main__":
        main(sys.argv)


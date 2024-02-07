1. For any new analysis: </b>
 - make a new analysis directory (e.g. mkdir Run_1)
 - run the command line prompt: make_stacking_run
 - this will copy all needed files. 

2. Specify inputs in inputs.yaml. </b>
 - This is the only file a user should have to modify (apart from running functions in the client code).
 
3. Uncomment functions inside the client code that you want to run. </b>

4. To run the code: 
 - To submit batch jobs: patch scripts are provided for both PBS and SLURM. 
 - To run from terminal: python client.py
</pre>

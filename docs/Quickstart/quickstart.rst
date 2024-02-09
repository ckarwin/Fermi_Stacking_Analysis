Quickstart Guide
----------------
For any new analysis:

1. Make new analysis directory: ``mkdir Run_1``, ``cd Run_1``
2. Run command line prompt ``make_stacking_run``, which will setup the directory with all needed files.
3. Specify inputs in ``inputs.yaml``.
4. Uncomment functions inside ``client.py`` that you want to run.
5. To run batch jobs: batch script templates are provided for both ``SLURM`` and ``PBS``.
6. To run from terminal: ``python client.py``

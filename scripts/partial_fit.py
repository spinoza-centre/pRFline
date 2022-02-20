#!/usr/bin/env python
#$ -cwd
#$ -j Y
#$ -V

import os
import sys, getopt
from pRFline import utils, fitting
from linescanning.utils import get_file_from_substring
opj = os.path.join

def main(argv):

    """partial_fit.py

    Preprocess the partial FOV functional files prior to pRF-fitting. Steps included are: NORDIC, motion
    correction, registration to FSnative, projection to surface, and high pass filtering with DCT-set.

    Parameters
    ----------
    -s <subject>        subject ID as used throughout the pipeline.
    -f <func_dir>       path to where the functional files live. We'll search for the "acq-3DEPI" 
                        tag. If multiple files are found, we'll run preprocessing for them all.
    -o <output_dir>     output directory; should be project root directory with <subject>/<ses-X>/
                        func.
    -l <log_dir>        directory that contains the "Screenshot"-directory to create the design matrix
    -v                  turn on verbose
    -g                  run model fitter with 'gauss' model instead of 'norm'. Will do grid+iter fit
                        regardless of model choice

    Returns
    ----------
    pRF-fitting results in the form of *npy-files (including parameters & predictions)

    Example
    ----------
    >>> python partial_fit.py -s sub-003 -f /mnt/export/data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -o /mnt/export/data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -l /data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/prf_design/sub-003_ses-0_task-pRF_run-0 -v

    >>> qsub -N prf_003 -pe smp 1 -wd /data1/projects/MicroFunc/Jurjen/programs/project_repos/pRFline/logs partial_fit.py -s sub-003 -f /mnt/export/data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -o /mnt/export/data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -l /data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/prf_design/sub-003_ses-0_task-pRF_run-0 -v
    """

    subject     = None
    func_dir    = None
    output_dir  = None
    log_dir     = None
    verbose     = False
    model       = "norm"

    try:
        opts = getopt.getopt(argv,"gvhs:n:f:d:o:l:",["subject=", "func_dir=", "output_dir=", "log_dir="])[0]
    except getopt.GetoptError:
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--subject"):
            subject = arg
        elif opt in ("-f", "--func_dir"):
            func_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-l", "--log_dir"):
            log_dir = arg
        elif opt in ("-v"):
            verbose = True
        elif opt in ("-g"):
            model = "gauss"       

    # this is a bit more informative than if len(argv) < 8..
    if subject == None:
        raise ValueError("Please specify a log-directory with the Screenshot-directory")            
    
    if func_dir == None:
        raise ValueError("Please specify the path to the functional files")

    if output_dir == None:
        raise ValueError("Please specify an output directory (e.g., derivatives/prf/<subject>/<ses-)")

    if log_dir == None:
        raise ValueError("Please specify a log-directory with the Screenshot-directory")

    func_files = get_file_from_substring(["hemi-LR", "bold.npy"], func_dir)

    model_fit = fitting.FitPartialFOV(subject,
                                      func_files=func_files,
                                      output_dir=output_dir,
                                      TR=1.111,
                                      log_dir=log_dir,
                                      stage='grid+iter',
                                      model=model,
                                      verbose=verbose)

    model_fit.fit()

if __name__ == "__main__":
    main(sys.argv[1:])
#!/usr/bin/env python
#$ -q long.q@jupiter
#$ -cwd
#$ -j Y
#$ -V

import os
import sys, getopt
from pRFline import fitting
from linescanning import utils
from bids import BIDSLayout
opj = os.path.join

def main(argv):

    """partial_fit.py

    Preprocess the partial FOV functional files prior to pRF-fitting. Steps included are: NORDIC, motion
    correction, registration to FSnative, projection to surface, and high pass filtering with DCT-set.

    Parameters
    ----------
    -b <bids_dir>       path pointing to derivatives-folder containing the input from `fMRIPrep` and the 
                        output for pRFs. Overrules <func_dir> and <output_dir>
    -f <func_dir>       path to where the functional files live. We'll search for the "acq-3DEPI" 
                        tag. If multiple files are found, we'll run preprocessing for them all.
    -o <output_dir>     output directory; should be project root directory with <subject>/<ses-X>/
                        func.
    -l <log_dir>        directory that contains the "Screenshot"-directory to create the design matrix
    --tr                set manual TR in seconds [default = 1.111s]
    -v|--verbose        turn on verbose
    -g|--gauss          run model fitter with 'gauss' model instead of 'norm'. Will do grid+iter fit
                        regardless of model choice
    --psc               use percent signal change as standardization [default]
    --zscore            use zscore as standardization
    --fsnative          fit in FSNative space
    --fsaverage         fit in FSAverage space
    --hrf               fit HRF with the pRFs
    --no-fit            only do preprocessing, exit before `fit()`-call

    Returns
    ----------
    pRF-fitting results in the form of *npy-files (including parameters & predictions)

    Example
    ----------
    >>> # set subject/session
    >>> subID=002
    >>> sesID=2
    >>> #
    >>> # set path
    >>> bids_dir=${DIR_DATA_DERIV}/fmriprep
    >>> log_dir=${DIR_DATA_SOURCE}/sub-${subID}/ses-${sesID}/sub-${subID}_ses-${sesID}_task-pRF_run-imgs
    >>> #
    >>> # submit to cluster or run locally with python
    >>> # job="python"
    >>> job="qsub -N pfov_${subID} -pe smp 5 -wd /data1/projects/MicroFunc/Jurjen/programs/project_repos/pRFline/logs"
    >>> #
    >>> ${job} partial_fit.py -s ${subID} -n ${sesID} -b ${bids_dir} -l ${log_dir} -v --fsnative # fit with fsnative
    """

    subject         = None
    session         = None
    bids_dir        = None
    func_dir        = None
    output_dir      = None
    log_dir         = None
    verbose         = False
    model           = "norm"
    fit_hrf         = False
    standardization = "psc"
    space           = 'func'
    TR              = 1.111
    rsq_thresh      = 0.05
    n_pix           = 100
    fit             = True

    try:
        opts = getopt.getopt(argv,"gvh:b:s:n:f:d:o:l:",["bids_dir=", "subject=", "session=", "func_dir=", "output_dir=", "log_dir=", "tr=", "hrf", "verbose", "fsnative", "fsaverage", "rsq=", "n_pix=", "no-fit"])[0]
    except getopt.GetoptError:
        print("ERROR while handling arguments.. Did you specify an 'illegal' argument..?")
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--subject"):
            subject = arg
        elif opt in ("-n", "--session"):
            session = arg            
        elif opt in ("-b", "--session"):
            bids_dir = arg            
        elif opt in ("-b", "--bids_dir"):
            bids_dir = arg           
        elif opt in ("-f", "--func_dir"):
            func_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-l", "--log_dir"):
            log_dir = arg
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-g", "--gauss"):
            model = "gauss"     
        elif opt in ("--zscore"):
            standardization = "zscore"
        elif opt in ("--psc"):
            standardization = "psc"
        elif opt in ("--fsaverage"):
            space = "fsaverage"
        elif opt in ("--fsnative"):
            space = "fsnative"     
        elif opt in ("--tr"):
            TR = float(arg)           
        elif opt in ("--rsq"):
            rsq_thresh = arg
        elif opt in ("--n_pix"):
            n_pix = arg   
        elif opt in ("--no-fit"):
            fit = False
        elif opt in ("--hrf"):
            fit_hrf = True            

    if len(argv) == 0:
        print(main.__doc__)
        sys.exit(1)
    
    # check first if we got a BIDS-directory. If not, check -f and -o flag
    if bids_dir == None:
        if func_dir == None:
            raise ValueError("Please specify the path to the functional files")

        if output_dir == None:
            raise ValueError("Please specify an output directory (e.g., derivatives/prf/<subject>/<ses-)")

        # look for files within func_dir. Ensures compatibility with BIDS-option below
        file_list = func_dir
    else:
        
        if subject == None:
            raise ValueError("Must have subject ID (only digits) for this option")

        if verbose:
            print(f"Reading BIDS-directory '{bids_dir}'")

        func_dir = bids_dir
        layout = BIDSLayout(func_dir, validate=False)
        file_list = layout.get(subject=subject, session=session, return_type='file')

        if verbose:
            print(f"BIDS layout: {layout}")
        
        # set output directory
        if session != None:
            base_dir = opj(f'sub-{subject}', f'ses-{session}')
        else:
            base_dir = f'sub-{subject}'

        # combine base with directory 
        output_dir = opj(os.path.dirname(bids_dir), 'prf', base_dir)

    if log_dir == None:
        raise ValueError("Please specify a log-directory with the Screenshot-directory")

    # make directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # fetch files depending on space
    if space == "func":
        # try to look for brain extracted file
        try:
            func_files = utils.get_file_from_substring(["masked_bold.nii.gz"], file_list, exclude="space-")
        except:
            func_files = utils.get_file_from_substring(["preproc_bold.nii.gz"], file_list, exclude="space-")
    else:
        func_files = utils.get_file_from_substring([f"space-{space}", "hemi-LR", "bold.func.npy"], file_list)

    model_fit = fitting.FitPartialFOV(func_files=func_files,
                                      output_dir=output_dir,
                                      TR=TR,
                                      log_dir=log_dir,
                                      stage='grid+iter',
                                      model=model,
                                      verbose=verbose,
                                      fit_hrf=fit_hrf,
                                      standardization=standardization,
                                      rsq_threshold=rsq_thresh,
                                      n_pix=n_pix)

    # fit
    if fit:
        model_fit.fit()

if __name__ == "__main__":
    main(sys.argv[1:])
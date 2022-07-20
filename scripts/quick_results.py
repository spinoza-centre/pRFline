#!/usr/bin/env python

import ast
import os
import sys, getopt
from pRFline import fitting
from linescanning import utils
opj = os.path.join

def main(argv):

    """quick_results.py

    Quick peak in the results of the pRF fit given a range of voxels. It will produce the plots for the gray matter ribbon containing the pRF-visualization as well as predicted timecourse + raw data. By default it will look for parameter estimates embedded in a pickle-file (*.pkl), a newly introduced output type from :class:`linescanning.prf.pRFmodelFitting`. This contains the data, predictions, and parameters. With the '--np' flag, you can tell the script to look for the old numpy-output instead.

    Parameters
    ----------
    -s <subject ID>     subject ID as used throughout the pipeline (e.g., 'sub-001')
    -n <session ID>     session ID you'd like to inspect (e.g., 2)
    -r <range>          voxel range to plot the predictions from (default = [359,365])
    -x|--xkcd           use xkcd-format for plotting
    --pdf               save individual figures as pdf
    --png               save individual figures as png
    --svg               save individual figures as svg [default]        
    --hrf               signifies that we have fitted the HRF during fitting, makes sure we select the correct parameter file
    --np                use numpy-output from :class:`linescanning.prf.pRFmodelFitting`, rather than the pickle-output

    Returns
    ----------
    pRF-fitting results in the form of *npy-files (including parameters & predictions)

    Example
    ----------
    >>> # fetch results of pRF-estimation with HRF
    >>> python inspect_results.py -s sub-002 -n 2 --hrf
    """

    subject     = None
    session     = None
    plot_range  = [359,365]
    plot_xkcd   = False
    ext         = "svg"
    fit_hrf     = False
    save        = False
    look_for    = "pkl"

    try:
        opts = getopt.getopt(argv,"xhs:n:r:",["sub=", "ses=", "range=", "pdf", "png", "svg", "hrf", "xkcd", "np"])[0]
    except getopt.GetoptError:
        print("ERROR while handling arguments.. Did you specify an 'illegal' argument..?")
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts: 
        if opt == '-h':
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--sub"):
            subject = arg            
        elif opt in ("-n", "--ses"):
            session = arg
        elif opt in ("-r", "--range"):
            plot_range = ast.literal_eval(arg)
        elif opt in ("-x", "--xkcd"):
            plot_xkcd = True
        elif opt in ("--pdf"):
            ext = "pdf"
        elif opt in ("--png"):
            ext = "png"
        elif opt in ("--svg"):
            ext = "svg"           
        elif opt in ("--np"):
            look_for = "npy"                            
    
    if len(argv) == 0:
        print(main.__doc__)
        sys.exit()

    # set directories
    project_dir = os.environ.get("DIR_PROJECTS")
    base_dir    = os.path.join(project_dir, 'VE-pRF')
    deriv_dir   = opj(base_dir, 'derivatives')
    prf_new     = opj(deriv_dir, 'prf', subject, f"ses-{session}")

    # fetch parameters with/without HRF
    if fit_hrf:
        pars = utils.get_file_from_substring(["model-norm", "stage-iter", "hrf-true", f"params.{look_for}"], prf_new)
    else:
        pars = utils.get_file_from_substring(["model-norm", "stage-iter", f"params.{look_for}"], prf_new, exclude="hrf-true")

    # check if list and if so, select first element
    if isinstance(pars, list):
        pars = pars[0]

    # plop into pRFResults object
    results = fitting.pRFResults(pars, verbose=True)

    # save stuff if extension is given
    if ext != None:
        save = True
        
    results.plot_prf_timecourse(vox_range=plot_range, xkcd=plot_xkcd, save=save, ext=ext)

if __name__ == "__main__":
    main(sys.argv[1:])

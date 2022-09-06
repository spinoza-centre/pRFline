#!/usr/bin/env python

import ast
import os
import sys, getopt
from pRFline import fitting
from linescanning import utils
from pathlib import Path
opj = os.path.join

def main(argv):

    """quick_results.py

    Quick peak in the results of the pRF fit given a range of voxels. It will produce the plots for the gray matter ribbon containing the pRF-visualization as well as predicted timecourse + raw data. By default it will look for parameter estimates embedded in a pickle-file (*.pkl), a newly introduced output type from :class:`linescanning.prf.pRFmodelFitting`. This contains the data, predictions, and parameters. With the '--np' flag, you can tell the script to look for the old numpy-output instead.

    Parameters
    ----------
    -s <subject ID>     subject ID as used throughout the pipeline (e.g., 'sub-001')
    -n <session ID>     session ID you'd like to inspect (e.g., 2)
    -r|--range <range>  voxel range to plot the predictions from (default = [359,365])
    -v|--vox <vox_nr>   plot single voxel. Mutually exclusive with `-r|--range`
    -x|--xkcd           use xkcd-format for plotting
    --pdf               save individual figures as pdf
    --png               save individual figures as png
    --svg               save individual figures as svg [default]        
    --hrf               signifies that we have fitted the HRF during fitting, makes sure we select the correct parameter file
    --np                use numpy-output from :class:`linescanning.prf.pRFmodelFitting`, rather than the pickle-output
    --epi               signifies that we should include *acq-3DEPI* in our search for pRF-parameters
    --no_overlap        do not create the overlap plot between target and line-scanning pRF
    --no_depth          do not create the plot with parameters across depth

    Returns
    ----------
    pRF-fitting results in the form of *npy-files (including parameters & predictions)

    Example
    ----------
    >>> # fetch results of pRF-estimation with HRF
    >>> python quick_results.py -s sub-002 -n 2 --hrf
    """

    subject     = None
    session     = None
    plot_range  = [359,365]
    plot_vox    = None
    plot_xkcd   = False
    ext         = "svg"
    fit_hrf     = False
    output_dir  = None
    save        = False
    look_for    = "pkl"
    acq         = None
    overlap     = True
    depth       = True

    try:
        opts = getopt.getopt(argv,"xhs:n:r:o:",["sub=", "ses=", "range=", "pdf", "png", "svg", "hrf", "xkcd", "np", "epi", "vox=", "out=", "no_depth", "no_overlap"])[0]
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
        elif opt in ("-o", "--out"):
            output_dir = arg            
        elif opt in ("-r", "--range"):
            plot_range = ast.literal_eval(arg)
        elif opt in ("-x", "--xkcd"):
            plot_xkcd = True
        elif opt in ("-v", "--vox"):
            plot_vox = int(arg)
        elif opt in ("--pdf"):
            ext = "pdf"
        elif opt in ("--png"):
            ext = "png"
        elif opt in ("--svg"):
            ext = "svg"           
        elif opt in ("--np"):
            look_for = "npy"
        elif opt in ("--epi"):
            acq = ["acq"]
        elif opt in ("--hrf"):
            fit_hrf = True
        elif opt in ("--no_overlap"):
            overlap = False
        elif opt in ("--no_depth"):
            depth = False
    
    if len(argv) == 0:
        print(main.__doc__)
        sys.exit()

    # set directories
    project_dir = os.environ.get("DIR_PROJECTS")
    base_dir    = os.path.join(project_dir, 'VE-pRF')
    deriv_dir   = opj(base_dir, 'derivatives')
    prf_new     = opj(deriv_dir, 'prf', subject, f"ses-{session}")

    # fetch parameters with/without HRF
    search_for = ["model-norm", "stage-iter", f"params.{look_for}"]
    if fit_hrf:
        search_for += ["hrf-true"]
        exclude = None
    else:
        exclude = "hrf-true"

    if isinstance(acq, list):
        search_for += acq

    # search for parameters    
    pars = utils.get_file_from_substring(search_for, prf_new, exclude=exclude)

    # plop into pRFResults object
    results = fitting.pRFResults(pars, verbose=True)

    # save stuff if extension is given
    if ext != None:
        save = True
        if output_dir == None:
            module_path = Path(fitting.__file__)
            output_dir = module_path.parents[1]/'results'/subject

            # make directory
            output_dir.mkdir(parents=True, exist_ok=True)

    # set range to None if we received single voxel
    if plot_vox != None:
        plot_range = None
    
    if overlap:
        results.plot_prf_timecourse(
            vox_nr=plot_vox,
            vox_range=plot_range, 
            xkcd=plot_xkcd, 
            save=save,
            save_dir=str(output_dir), 
            ext=ext)

    if depth:
        measures = 'all'
        measures = ['prf_size', 'r2', 'size ratio']
        results.plot_depth(
            vox_range=plot_range, 
            xkcd=plot_xkcd, 
            save=save,
            save_dir=str(output_dir), 
            measures=measures,
            ext=ext,
            font_size=20,
            label_size=14,
            set_xlim_zero=False,
            cmap="inferno")

if __name__ == "__main__":
    main(sys.argv[1:])

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
    --norm              use normalization parameters [default]
    --gauss             use Gaussian parameters
    --dog               use Difference-of-Gaussian parameters
    --css               use CSS parameters
    --avg               read in the average across GM files; will produce the overlap-with-target figure
    --ribbon            read in the ribbon files; will produce the depth-figure; will automatically overwrite `plot_range` to whatever format the data is
    --targ2line         Plot prediction of target pRF in response to line-scanning design. This option is only available with `--avg`.

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
    model       = "norm"
    exclude     = "vox-"
    data_type   = None
    targ2line   = False

    try:
        opts = getopt.getopt(argv,"xhs:n:r:o:",["sub=", "ses=", "range=", "pdf", "png", "svg", "hrf", "xkcd", "np", "epi", "vox=", "out=", "no_depth", "no_overlap", "norm", "css", "gauss", "dog", "ribbon", "avg", "targ2line"])[0]
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
        elif opt in ("--gauss"):
            model = "gauss"
        elif opt in ("--norm"):
            model = "norm"
        elif opt in ("--css"):
            model = "css"
        elif opt in ("--dog"):
            model = "dog"
        elif opt in ("--ribbon"):
            data_type = "vox-ribbon"
            exclude = None
        elif opt in ("--avg"):
            data_type = "vox-avg"
            exclude = None
            plot_vox = 0
            depth = False
        elif opt in ("--targ2line"):
            targ2line = True

    if len(argv) == 0:
        print(main.__doc__)
        sys.exit()

    # set directories
    project_dir = os.environ.get("DIR_PROJECTS")
    base_dir    = os.path.join(project_dir, 'VE-pRF')
    deriv_dir   = opj(base_dir, 'derivatives')
    prf_new     = opj(deriv_dir, 'prf', subject, f"ses-{session}")

    # fetch parameters with/without HRF
    search_for = [
        f"model-{model}", 
        "stage-iter", 
        data_type,
        f"params.{look_for}"]

    if isinstance(acq, list):
        search_for += acq

    # search for parameters    
    pars = utils.get_file_from_substring(search_for, prf_new, exclude=exclude)

    if isinstance(pars, list):
        if fit_hrf:
            pars = utils.get_file_from_substring(['hrf-true'], pars)
        else:
            pars = utils.get_file_from_substring(search_for, pars, exclude=['hrf-true'])
    
    if isinstance(pars, list):
        raise ValueError(f"Found multiple files ({len(pars)}: {pars}")

    # plop into pRFResults object
    results = fitting.pRFResults(
        pars, 
        verbose=True,
        targ2line=targ2line)

    # save stuff if extension is given
    if ext != None:
        save = True
        if output_dir == None:
            module_path = Path(fitting.__file__)
            output_dir = module_path.parents[1]/'results'/subject

            # make directory
            output_dir.mkdir(parents=True, exist_ok=True)

    # set range to whatever the data is
    if "ribbon" in data_type:
        plot_range = [0,results.data.shape[-1]]

    # set range to None if we received single voxel
    if isinstance(plot_vox, int):
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
        if isinstance(plot_vox, int) or "avg" in data_type:
            raise ValueError("Cannot use this plotting type with the average timecourse; use --ribbon")

        measures = 'all' # x,y,prf_size,prf_ampl,bold_bsl,neur_bsl,surr_ampl,surr_size,surr_bsl,A,B,C,D,ratio (B/D),r2,size ratio,suppression index

        if model == "norm":
            measures = ['prf_size', 'r2', 'size ratio', 'suppression index']
        else:
            measures = ['prf_size', 'r2']
            
        results.plot_depth(
            vox_range=plot_range, 
            xkcd=plot_xkcd, 
            save=save,
            save_dir=str(output_dir), 
            measures=measures,
            ext=ext,
            font_size=30,
            label_size=20,
            set_xlim_zero=False,
            cmap="inferno")

if __name__ == "__main__":
    main(sys.argv[1:])

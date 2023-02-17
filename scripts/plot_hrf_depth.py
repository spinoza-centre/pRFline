#!/usr/bin/env python
#$ -q long.q
#$ -cwd
#$ -j Y
#$ -o ../logs
#$ -V
#$ -N plot_hrf_depth

import os
import sys
import getopt
import pRFline
from linescanning import utils
from pRFline import figures
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """plot_hrf_depth.py

Creates the figure showing the timecourses for white matter and pial voxels, positional stability across depth, and HRF'ey profiles

Parameters
----------
--fwhm  
Example
----------
>>> ./plot_hrf_depth.py
    """

    verbose = True
    inset = "fwhm"

    try:
        opts = getopt.getopt(argv,"h:",["help", "fwhm", "mag", "ttp"])[0]
    except getopt.GetoptError:
        print("ERROR while handling arguments.. Did you specify an 'illegal' argument..?", flush=True)
        print(main.__doc__, flush=True)
        sys.exit(2)

    for opt, arg in opts: 
        if opt in ("-h", "--help"):
            print(main.__doc__)
            sys.exit()
        elif opt in ("--fwhm"):
            inset = "fwhm"
        elif opt in ("--mag"):
            inset = "mag"
        elif opt in ("--ttp"):
            inset = "ttp"   

    utils.verbose("\nplot_hrf_depth.py\n", verbose)

    # set defaults
    fig_dir = opj(opd(opd(pRFline.__file__)), "figures")
    data_dir = opj(opd(opd(pRFline.__file__)), "data")

    # check if we have full parameter file; saves time
    params_fn = opj(data_dir, f"sub-all_model-gauss_desc-full_params.csv")
    if not os.path.exists(params_fn):
        params_fn = None

    im5 = figures.DepthHRF(
        full_dict=params_fn, 
        hrf_csv=opj(data_dir, f"sub-all_model-gauss_desc-hrf_across_depth.csv"),
        metric_csv=opj(data_dir, "sub-all_model-gauss_desc-hrf_metrics.csv"),
        verbose=True)

    im5.compile_depth_figure(
        insets=inset,
        save_as=opj(fig_dir, f"sub-all_model-gauss_fig-6_desc-hrf_depth")
    )        

if __name__ == "__main__":
    main(sys.argv[1:])

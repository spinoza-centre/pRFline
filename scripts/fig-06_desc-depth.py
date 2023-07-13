#!/usr/bin/env python
#$ -q long.q
#$ -cwd
#$ -j Y
#$ -o ../logs
#$ -V
#$ -N f06_depth

import os
import sys
import getopt
import pRFline
from linescanning import utils
import numpy as np
from pRFline import figures
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """fig-06_desc-depth.py

Creates the figure showing a couple depth-dependent results

(A) shows the anatomical slice in FreeSurfer space with the beam and ribbon images overlaid. Note that these are imshows of actual nifti-images. In (B) we have the raw (thin lines) and predicted (thick lines) time courses for a superficial (red) and deep (blue) voxel. In (C) we show the position estimates across depth in color, with the average across the ribbon voxels in black. (D) shows the profile of the first bar pass shown in B. This is annotated with the grey boxes below. In (E) and (F) we have the magnitude and variance explained, respectively, as a function of depth

Parameters
----------
-s|--subject        use curvature/distance images from this subject. Default = 'sub-003'
--gauss             fit gaussian model (default)
--norm              fit normalization model
--css               fit CSS model
--dog               fit DoG model
--raw               plot raw timecourses rather than filtered ones. Should be used for the figure!

Example
----------
>>> ./fig-06_desc-depth.py
>>> ./fig-06_desc-depth.py --raw
    """

    subject = "sub-003"
    verbose = True
    model = "gauss"
    raw = False

    try:
        opts = getopt.getopt(argv,"h:s:",["help", "norm", "gauss", "dog", "css", "raw"])[0]
    except getopt.GetoptError:
        print("ERROR while handling arguments.. Did you specify an 'illegal' argument..?", flush=True)
        print(main.__doc__, flush=True)
        sys.exit(2)

    for opt, arg in opts: 
        if opt in ("-h", "--help"):
            print(main.__doc__)
            sys.exit()    
        elif opt in ("-s","--subject"):
            subject = arg
        elif opt in ("--norm"):
            model = "norm"
        elif opt in ("--gauss"):
            model = "gauss"
        elif opt in ("--dog"):
            model = "dog"
        elif opt in ("--css"):
            model = "css"        
        elif opt in ("--raw"):
            raw = True

    utils.verbose("\nfig-06_desc-depth.py\n", verbose)

    # set defaults
    fig_dir = opj(opd(opd(pRFline.__file__)), "figures")
    data_dir = opj(opd(opd(pRFline.__file__)), "data")
    base_dir = opj(os.environ.get("DIR_PROJECTS"), "VE-pRF")

    # check if we have full parameter file; saves time
    params_fn = opj(data_dir, f"sub-all_model-{model}_desc-full_params.csv")
    if not os.path.exists(params_fn):
        params_fn = None

    # initialize class
    im6 = figures.DepthHRF(
        full_dict=params_fn, 
        deriv=opj(base_dir, 'derivatives'),
        hrf_csv=opj(data_dir, f"sub-all_model-gauss_desc-hrf_across_depth.csv"),
        metric_csv=opj(data_dir, "sub-all_model-gauss_desc-hrf_metrics.csv"),
        verbose=True,
        label_size=18,
        subject=subject,
        code=3)
    
    # read raw data
    if raw:
        ses = im6.subj_obj.get_session(im6.subject)
        fname = opj(
            base_dir,
            "derivatives",
            "prf",
            im6.subject,
            f"ses-{ses}",
            f"{im6.subject}_ses-{ses}_task-pRF_desc-data_for_fitter_raw.npy")
        
        # check if file exists, otherwise prompt command to create it
        if os.path.exists(fname):
            raw_tcs = np.load(fname)
        else:
            raise FileNotFoundError(f"Could not find raw data. Create with './line_fit -s {subject} -i 2 --verbose --save_tc {fname}'")
        
        # get voxel IDs of ribbon
        rib_range = im6.subj_obj.get_ribbon(im6.subject)
        use_ranges = [rib_range[0],rib_range[1]-1]

        # get the predictions
        pial_raw = raw_tcs[:,use_ranges[0]]
        wm_raw = raw_tcs[:,use_ranges[1]]

        # store in list
        plot_list = [
            pial_raw+abs(im6.pial_pred[0]),
            im6.pial_pred+abs(im6.pial_pred[0]),
            wm_raw+abs(im6.wm_pred[0]),
            im6.wm_pred+abs(im6.wm_pred[0])
        ]

    else:
        # use internally stored time courses; these are low-passed!
        plot_list = None

    # compile figure
    im6.compile_depth_figure2(
        # general plotting kwargs
        plot_kwargs={
            "font_size": 25,
            "label_size": 20,
            "y_dec": 1
        },
        # zoom options for (A)
        wb_kwargs={
            "vmax": 1000,
            "zoom": [176,352,88,264]
        },
        # list with timecourses for (B)
        tcs=plot_list,
        # annotation kwargs for visual field delineation in (C)
        scatter_kwargs={
            "annot_pos": {
                "pos": [(-0.1,0),(-0.1,0.96),(0.96,0.96)],
                "txt": ["-1","0","2"]},
            "extent": [-0.1,2,-1,0.1],
        },
        # kwargs for axvspan object in B/D
        tc_kwargs={
            "start": 0,
            "end": 30,
            "xlim_left": 5
        },
        # save
        save_as=opj(fig_dir, f"sub-all_model-gauss_fig-6_desc-hrf_depth")
    )

if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python

import os
import sys
import getopt
import pRFline
from pRFline.utils import SubjectsDict
from pRFline import surface
from linescanning import utils, pycortex
import pandas as pd
import time
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """dist_on_surf.py

Project the difference of all V1-pRFs to the average line-pRF to the surface.

Parameters
----------
-s|--subject        process specific subject. Default = "all"
-n|--session        process specific session. Default = 2, but I know that for sub-005 it's '3'
-v|--no_verbose     turn off verbose (best to have verbose on by default)
-k|--kernel         smoothing kernel. Default = 1
-i|--n_iter         number of iterations to do when smoothing. Default = 1
--lh|--rh           hemisphere to process (default = "lh")
--lp                points to a different location for line-estimates
--gauss             fit gaussian model (default)
--norm              fit normalization model
--css               fit CSS model
--dog               fit DoG model
--webshow           open pycortex in browser to create images (also saves them). Default is False
--smooth            smooth the distance maps on the surface. Default = False
--no_verbose        turn off verbose

Returns
----------
Opens a webshow from pycortex to save out the figure, as well as pandas dataframe with the distance for each subject

Example
----------
>>> ./dist_on_surf.py
>>> ./dist_on_surf.py -s sub-001
>>> ./dist_on_surf.py --dog
>>> ./dist_on_surf.py --smooth --kernel 2 
    """

    verbose = True
    model = "gauss"
    subject = "all"
    filt_strat = "hp"
    hemi = "lh"
    webshow = False
    n_iter = 1
    kernel = 1
    smooth = False

    try:
        opts = getopt.getopt(argv,"hm:v:k:i:",["help", "subject=", "session=", "no_verbose", "norm", "gauss", "dog", "css", "lp", "lh", "rh", "webshow", "sm", "smooth", "kernel", "iter", "no_verbose"])[0]
    except getopt.GetoptError:
        print("ERROR while handling arguments.. Did you specify an 'illegal' argument..?", flush=True)
        print(main.__doc__, flush=True)
        sys.exit(2)

    for opt, arg in opts: 
        if opt in ("-h", "--help"):
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--subject"):
            subject = arg
        elif opt in ("-v", "--no_verbose"):
            verbose = False
        elif opt in ("-k", "--kernel"):
            kernel = arg
        elif opt in ("-i", "--n_iter"):
            n_iter = arg            
        elif opt in ("--norm"):
            model = "norm"
        elif opt in ("--gauss"):
            model = "gauss"
        elif opt in ("--dog"):
            model = "dog"
        elif opt in ("--css"):
            model = "css"
        elif opt in ("--lp"):
            filt_strat = "lp"
        elif opt in ("--lh"):
            hemi = "lh" 
        elif opt in ("--rh"):
            hemi = "rh"   
        elif opt in ("--webshow"):
            webshow = True
        elif opt in ("--sm", "--smooth"):
            smooth = True
        elif opt in ("--no_verbose"):
            verbose = False            
        

    utils.verbose("\ndist_on_surf.py", verbose)

    # set defaults
    base_dir    = "/data1/projects/MicroFunc/Jurjen/projects/VE-pRF"
    design_dir  = opj(opd(opd(pRFline.__file__)), "data")
    fig_dir     = opj(opd(opd(pRFline.__file__)), "results")

    if filt_strat == "lp":
        add = "_lp3"
        design_dir += add

    # fetch subject dictionary from pRFline.utils.SubjectsDict
    subj_obj = SubjectsDict()
    dict_data = subj_obj.dict_data

    if subject == "all":
        process_list = list(dict_data.keys())
    else:
        process_list = [subject]

    distsurf = {}
    for ii in ["subject","geodesic","euclidian","dva"]:
        distsurf[ii] = []

    for subject in process_list:

        utils.verbose(f"\n**************************************** Processing {subject} ***************************************", verbose)
        
        # read subject-specific session from dictionary
        ses = subj_obj.get_session(subject)
        data_output = opj(design_dir, subject)
        
        # prf directory
        prf_dir = opj(
            base_dir,
            "derivatives",
            "prf",
            subject)

        # find line-estimates
        prf_line = opj(prf_dir, f"ses-{ses}", f"{subject}_ses-{ses}_task-pRF_run-avg_vox-avg_model-{model}_stage-iter_desc-prf_params.pkl")
        
        # only have norm/gauss parameters for V1; so in case of DoG/CSS, use gaussian parameters for comparison
        if model == "norm":
            use_md = "norm"
        else:
            use_md = "gauss"

        # find EPI estimates
        prf_epi = opj(prf_dir, "ses-1", f"{subject}_ses-1_task-2R_roi-V1_model-{use_md}_stage-iter_desc-prf_params.pkl")

        # put everything in class
        if os.path.exists(prf_line) and os.path.exists(prf_epi):

            # initialize object
            sg = surface.DistSurf(
                subject=subject,
                hemi=hemi,
                verbose=verbose
            )

            # read the pRF estimate files
            sg.read_files(
                epi=prf_epi,
                line=prf_line
            )

            # find distances (smooth if requested)
            sg.find_distance(
                smooth=smooth, 
                kernel=kernel,
                iterations=n_iter)

            # append to list
            distsurf["subject"].append(subject)
            distsurf["geodesic"].append(sg.dist_geodesic)
            distsurf["euclidian"].append(sg.dist_euclidian)
            distsurf["dva"].append(sg.dva_dist)

            if webshow:
                # make the vertex objects
                sg.make_vertex(one_hemi=True)

                # create data dict
                data_d = {
                    "distance": sg.dist_v,
                    "targ-match": sg.target_matched_v
                }

                # initialize saving object (also opens the webviewers)
                pyc_save = pycortex.SavePycortexViews(
                    data_d,
                    subject=subject,
                    fig_dir=opj(fig_dir,subject),
                    zoom=True)

                time.sleep(3)
                pyc_save.save_all()
        else:
            raise FileNotFoundError(f"Could not find '{prf_line}' or '{prf_epi}'")

    if len(distsurf) > 0:
        if smooth:
            add_str = f"smooth-true_kernel-{kernel}_iter-{n_iter}"
        else:
            add_str = "smooth-false"

        fname = opj(fig_dir, f"model-{model}_{add_str}_desc-dist_on_surf.csv")
        utils.verbose(f"Writing {fname}", verbose)
        df_surf = pd.DataFrame(distsurf)
        df_surf["ix"] = 0
        df_surf.to_csv(fname)

if __name__ == "__main__":
    main(sys.argv[1:])

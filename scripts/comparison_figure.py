#!/usr/bin/env python

import os
import sys
import getopt
import pRFline
from linescanning import utils
from pRFline.utils import SubjectsDict
from pRFline import figures
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """main_figure.py

Produces the main figure containing details on the overlap (subject-specific as well as normalized), r2 measures, and measures of distances on the surface.

Parameters
----------
-s|--subject        use curvature/distance images from this subject. Default = 'sub-002'
--gauss             fit gaussian model (default)
--norm              fit normalization model
--css               fit CSS model
--dog               fit DoG model

Example
----------
>>> ./main_figure.py
>>> ./main_figure.py --dog
    """

    subject = "sub-005"
    verbose = True
    model = "gauss"

    try:
        opts = getopt.getopt(argv,"h:",["help", "norm", "gauss", "dog", "css"])[0]
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

    utils.verbose("\ncomparison_figure.py", verbose)

    # set defaults
    fig_dir = opj(opd(opd(pRFline.__file__)), "results")

    # fetch subject dictionary from pRFline.utils.SubjectsDict
    cmap_subj = "Set2"

    # define a bunch of files for panel AB
    img1 = opj(
        fig_dir, 
        subject, 
        f"{subject}_model-{model}_smooth-false_desc-distance_pivot.png")

    img2 = opj(
        fig_dir, 
        subject, 
        f"{subject}_model-{model}_smooth-true_kernel-1_iter-1_desc-distance_pivot.png")        

    # and panel EFG
    csv_file = opj(
        fig_dir, 
        f"sub-all_model-{model}_smooth-true_kernel-1_iter-1_desc-dist_on_surf.csv")

    # check if the files exist
    for ii in [img1,img2,csv_file]:
        if not os.path.exists(ii):
            raise FileNotFoundError(f"Could not find '{ii}'.")

    # check if we have full parameter file; saves time
    params_fn = opj(fig_dir, f"sub-all_model-{model}_desc-full_params.csv")
    if not os.path.exists(params_fn):
        params_fn = None
    
    # initialize class
    im = figures.CrossBankFigure(
        full_dict=params_fn,
        model=model,
        cmap=cmap_subj,
        verbose=True,
        targ_match_colors=["r","b"]
    )

    # compile the figure
    im.plot_comparison(
        img1=img1,
        img2=img2,
        targ=(993,3397),
        match1=(920,3480),
        match2=(1073,3521),
        cbar_inset=[-0.15,0.1,0.02,0.8],
        save_as=opj(os.path.dirname(params_fn),f"sub-all_model-{model}_desc-figure2"),
        csv_file=opj(fig_dir, f"{subject}_model-{model}_desc-smoothing_comparison.csv"),
        wspace=0.6,
        inset=[0.7,-0.2,0.7,0.7],
        figsize=(24,5),
        leg_anchor=(3.6,1)
    )

if __name__ == "__main__":
    main(sys.argv[1:])

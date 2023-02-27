#!/usr/bin/env python
#$ -q short.q
#$ -cwd
#$ -j Y
#$ -o ../logs
#$ -V
#$ -N f06_comparison_figure

import os
import sys
import getopt
import pRFline
from linescanning import utils
from pRFline import figures
from pRFline.utils import SubjectsDict
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """fig-06_desc-comparison.py

Plots the effect of smoothing on distance measures.

Parameters
----------
-s|--subject        use curvature/distance images from this subject. Default = 'sub-005'
--gauss             fit gaussian model (default)
--norm              fit normalization model
--css               fit CSS model
--dog               fit DoG model

Example
----------
>>> ./fig-06_desc-comparison.py
>>> ./fig-06_desc-comparison.py -s sub-003 
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

    utils.verbose("\nfig-06_desc-comparison.py", verbose)

    # set defaults
    results_dir = opj(opd(opd(pRFline.__file__)), "results")
    fig_dir = opj(opd(opd(pRFline.__file__)), "figures")
    data_dir = opj(opd(opd(pRFline.__file__)), "data")

    subj_obj = SubjectsDict()
    ses = subj_obj.get_session(subject)

    # fetch subject dictionary from pRFline.utils.SubjectsDict
    cmap_subj = "Set2"

    # define a bunch of files for panel AB
    img1 = opj(
        results_dir, 
        subject, 
        f"{subject}_ses-{ses}_model-{model}_smooth-false_desc-distance_pivot.png")

    img2 = opj(
        results_dir, 
        subject, 
        f"{subject}_ses-{ses}_model-{model}_smooth-true_kernel-1_iter-1_desc-distance_pivot.png")        

    # and panel EFG
    csv_file = opj(
        data_dir, 
        f"sub-all_model-{model}_smooth-true_kernel-1_iter-1_desc-dist_on_surf.csv")

    # check if the files exist
    for ii in [img1,img2,csv_file]:
        if not os.path.exists(ii):
            raise FileNotFoundError(f"Could not find '{ii}'.")

    # check if we have full parameter file; saves time
    params_fn = opj(data_dir, f"sub-all_model-{model}_desc-full_params.csv")
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
        save_as=opj(fig_dir, f"sub-all_model-{model}_fig-6_desc-effect_smoothing"),
        csv_file=opj(data_dir, subject, f"{subject}_ses-{ses}_model-{model}_desc-smoothing_comparison.csv"),
        wspace=0.6,
        inset=[0.7,-0.2,0.7,0.7],
        figsize=(24,5),
        leg_anchor=(3.6,1)
    )

if __name__ == "__main__":
    main(sys.argv[1:])

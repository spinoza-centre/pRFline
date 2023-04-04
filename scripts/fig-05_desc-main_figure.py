#!/usr/bin/env python
#$ -q long.q
#$ -cwd
#$ -j Y
#$ -o ../logs
#$ -V
#$ -N f05_main_figure

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

    """fig-05_desc-main_figure.py

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
>>> ./fig-05_desc-main_figure.py
>>> ./fig-05_desc-main_figure.py --dog
    """

    subject = "sub-002"
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

    utils.verbose("\nfig-05_desc-main_figure.py\n", verbose)

    # set defaults
    results_dir = opj(opd(opd(pRFline.__file__)), "results")
    data_dir = opj(opd(opd(pRFline.__file__)), "data")
    fig_dir = opj(opd(opd(pRFline.__file__)), "figures")

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # fetch subject dictionary from pRFline.utils.SubjectsDict
    cmap_subj = "Set2"

    # fetch subject dictionary from pRFline.utils.SubjectsDict
    subj_obj = SubjectsDict()
    ses = subj_obj.get_session(subject)

    # define a bunch of files for panel D
    base = f"{subject}_ses-{ses}"
    img_dist = opj(
        results_dir, 
        subject, 
        f"{base}_model-{model}_smooth-true_kernel-1_iter-1_desc-distance.png")

    # and panel EFG
    csv_file = opj(
        data_dir, 
        f"sub-all_model-{model}_smooth-true_kernel-1_iter-1_desc-dist_on_surf.csv")

    # check if the files exist
    for ii in [img_dist,csv_file]:
        if not os.path.exists(ii):
            raise FileNotFoundError(f"Could not find '{ii}'. Please create it with ./dist_on_surf.py")

    # check if we have full parameter file; saves time
    params_fn = opj(data_dir, f"sub-all_model-{model}_desc-full_params.csv")
    if not os.path.exists(params_fn):
        params_fn = None
    
    # initialize class
    im = figures.MainFigure(
        full_dict=params_fn,
        model=model,
        cmap=cmap_subj,
        verbose=True,
        targ_match_colors=["r","b"],
        label_size=20,
        font_size=26
    )

    # compile the figure
    im.compile_figure(
        img_dist=img_dist,
        csv_file=csv_file,
        save_as=opj(fig_dir, f"sub-all_model-{model}_fig-5_desc-functional_precision"),
        coord_targ=(1594,3172),
        coord_closest=(1594,3205),
        include=["euclidean","geodesic"],
        inset_axis=[0.6,-0.25,0.7,0.7],
        inset_extent=[1000,2600,2400,3400],
        cbar_inset=[0.1,1.1,0.8,0.05],
        txt_pos1=(-50,15),
        txt_pos2=(0,-25),
        flip_ticks=False,
        flip_label=True,
        annot_size=32)

if __name__ == "__main__":
    main(sys.argv[1:])

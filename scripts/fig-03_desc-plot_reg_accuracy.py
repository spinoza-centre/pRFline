#!/usr/bin/env python
#$ -q long.q
#$ -cwd
#$ -j Y
#$ -o ../logs
#$ -V
#$ -N f03_plot_reg_accuracy

import os
import sys
import getopt
import pRFline
from linescanning import utils
from pRFline import figures
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """fig-03_desc-plot_reg_accuracy.py

Creates the figure showing the registration cascade, as well as the registration accuracy distribution between ses-1 and ses-2. 

Parameters
----------
--gauss             fit gaussian model (default)
--norm              fit normalization model
--css               fit CSS model
--dog               fit DoG model
--no_verbose        turn off verbose (though there's not much verbose going on)

Example
----------
>>> ./fig-03_desc-plot_reg_accuracy.py
    """

    verbose = True
    model = "gauss"

    try:
        opts = getopt.getopt(argv,"h:",["help","gauss","norm","css","dog","no_verbose"])[0]
    except getopt.GetoptError:
        print("ERROR while handling arguments.. Did you specify an 'illegal' argument..?", flush=True)
        print(main.__doc__, flush=True)
        sys.exit(2)

    for opt, arg in opts: 
        if opt in ("-h", "--help"):
            print(main.__doc__)
            sys.exit()
        elif opt in ("--norm"):
            model = "norm"
        elif opt in ("--gauss"):
            model = "gauss"
        elif opt in ("--dog"):
            model = "dog"
        elif opt in ("--css"):
            model = "css"       
        elif opt in ("--no_verbose"):
            verbose = False

    utils.verbose("\nfig-03_desc-plot_reg_accuracy.py\n", verbose)

    # set defaults
    fig_dir = opj(opd(opd(pRFline.__file__)), "figures")
    data_dir = opj(opd(opd(pRFline.__file__)), "data")

    # check if we have full parameter file; saves time
    params_fn = opj(data_dir, f"sub-all_model-{model}_desc-full_params.csv")
    if not os.path.exists(params_fn):
        params_fn = None

    im4 = figures.AnatomicalPrecision(
        full_dict=params_fn,
        reg_csv=opj(data_dir, "sub-all_desc-registration.csv"),
        moco_csv=opj(data_dir, "sub-all_model-gauss_desc-slice_motion.csv"),
        model="gauss",
        verbose=True,
        annot_size=32,
        label_size=20,
        font_size=24
    )    
    
    im4.compile_reg_figure(
        inset_axis=[0.7,-2.3,2.5,2.5],
        save_as=opj(fig_dir, f"sub-all_model-{model}_fig-3_desc-anatomical_precision"))

if __name__ == "__main__":
    main(sys.argv[1:])

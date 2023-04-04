#!/usr/bin/env python
#$ -q long.q
#$ -cwd
#$ -j Y
#$ -o ../logs
#$ -V
#$ -N f04_predict_from_epi

import os
import sys
import getopt
import pRFline
from linescanning import utils
from pRFline import figures
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """fig-04_desc-predict_from_epi.py

Creates the figure showing the predicted timecourse of the target pRF given the line-scanning design matrix.

In A) we have the the target pRF (orange) with the line pRF (green); in B) the predicted timecourses for both (I performed an additional GLM on the EPI prediction on the line-scanning data to deal with amplitude differences), and in C) the variance explained of the line-pRFs, (GLM-fitted) EPI predictions, and the r2 assuming the design matrix as block design; ones whenever the stimulus is on the screen. This deals with the positional bias from the predicted EPI pRF and provides a more fair estimate of out-of-experiment r2.

Parameters
----------
-s|--subject        use curvature/distance images from this subject. Default = 'sub-005'
--gauss             fit gaussian model (default)
--norm              fit normalization model
--css               fit CSS model
--dog               fit DoG model

Example
----------
>>> ./fig-04_desc-predict_from_epi.py
>>> ./fig-04_desc-predict_from_epi.py --dog
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

    utils.verbose("\nfig-04_desc-predict_from_epi.py\n", verbose)

    # set defaults
    fig_dir = opj(opd(opd(pRFline.__file__)), "figures")
    data_dir = opj(opd(opd(pRFline.__file__)), "data")

    # check if we have full parameter file; saves time
    params_fn = opj(data_dir, f"sub-all_model-{model}_desc-full_params.csv")
    if not os.path.exists(params_fn):
        params_fn = None

    # check if we have h5 file for figure; saves time
    h5_file = opj(data_dir, f"sub-all_model-{model}_desc-predict_from_epi.h5")
    
    # initialize class
    im3 = figures.WholeBrainToLine(
        full_dict=params_fn,
        h5_file=h5_file,
        model="gauss",
        verbose=True,
        label_size=20,
        font_size=24
    )

    im3.plot_predictions(
        subject=subject, 
        exclude_line=False, 
        posthoc=True,
        save_as=opj(fig_dir, f"sub-all_model-{model}_fig-4_desc-predict_from_epi"),
        ast_frac=0.05,
        annot_size=32)

if __name__ == "__main__":
    main(sys.argv[1:])
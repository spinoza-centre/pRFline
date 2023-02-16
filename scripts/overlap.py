#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V
#$ -q long.q@jupiter
#$ -o /data1/projects/MicroFunc/Jurjen/programs/project_repos/pRFline/logs

from datetime import datetime
import os
import sys
import getopt
import pRFline
import numpy as np
from pRFline.utils import SubjectsDict, read_subject_data
from linescanning import (
    utils,
    prf
)
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """overlap.py

fit individual runs of 2D-EPI data & avg of line-scanning data

Parameters
----------
-s|--subject        process specific subject. Default = "all"
-n|--session        process specific session. Default = 2, but I know that for sub-005 it's '3'
-v|--no_verbose     turn off verbose (best to have verbose on by default)
--lp                low-pass filter the data (savitsky-golay filter, window length=11, order=3)
--gauss             fit gaussian model (default)
--norm              fit normalization model
--css               fit CSS model
--dog               fit DoG model
--fit_baseline      don't fix bold baseline (default = False)
--fit_position      don't fix position across depth (default = False)
--plot              make figure containing overlap of line-pRF & 2D-EPI pRF in visual space + bar plots regarding distance from 2D-EPI pRF & variance explained
--ds|--downsample   downsample line-scanning data to epi-resolution
--skip_epi          do not process the whole-brain EPI files
--skip_lines        do not process the line-scanning data
--epi_avg           only do EPI average, not individual runs

Returns
----------
A bunch of pkl-files that can be used in conjuction with `group_overlap.ipynb`

Example
----------
>>> # fit individual runs of 2D-EPI data & avg of line-scanning data
>>> python overlap.py
>>> model="gauss"; qsub -N fig_${model} overlap.py --plot --${model}
>>> model="norm"; qsub -N fig_${model} overlap.py --plot --${model}
    """

    verbose = True
    fix_bold = True
    fix_pos = True
    model = "gauss"
    subject = "all"
    plot = False
    fit_nbr = False
    overwrite = False
    skip_lines = False
    skip_epi = False
    downsample = False
    filt_strat = "hp"
    window = None
    poly = None
    ses = 2

    try:
        opts = getopt.getopt(argv,"hov:m:f:s:r:n:",["help", "run=", "subject=", "session=", "no_verbose", "fit_baseline", "fit_position", "norm", "gauss", "dog", "css", "plot", "nbr", "ow", "overwrite", "skip_epi", "skip_lines", "ds", "downsample", "epi_avg" ,"lp"])[0]
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
        elif opt in ("--norm"):
            model = "norm"
        elif opt in ("--gauss"):
            model = "gauss"
        elif opt in ("--dog"):
            model = "dog"
        elif opt in ("--css"):
            model = "css"
        elif opt in ("--fit_baseline"):
            fix_bold = False
        elif opt in ("--fit_position"):
            fix_pos = False
        elif opt in ("--plot"):
            plot = True     
        elif opt in ("--nbr"):
            fit_nbr = True          
        elif opt in ("-o", "--ow", "--overwrite"):
            overwrite = True
        elif opt in ("--skip_lines"):
            skip_lines = True
        elif opt in ("--skip_epi"):
            skip_epi = True
        elif opt in ("--ds", "--downsample"):
            downsample = True
        elif opt in ("--lp"):
            filt_strat = "lp"
            window = 11
            poly = 3

    # set defaults
    task        = "task-pRF"
    base_dir    = "/data1/projects/MicroFunc/Jurjen/projects/VE-pRF"
    design_dir  = opj(opd(opd(pRFline.__file__)), "data")
    fig_dir     = opj(opd(opd(pRFline.__file__)), "results")

    if filt_strat == "lp":
        add = "_lp3"
        design_dir += add
        fig_dir += add


    # fetch subject dictionary from pRFline.utils.SubjectsDict
    subj_obj = SubjectsDict()
    dict_data = subj_obj.dict_data

    if subject == "all":
        process_list = list(dict_data.keys())
    else:
        process_list = [subject]

    utils.verbose("overlap.py", verbose)

    full_dict = {}
    for subject in process_list:
        
        dd = read_subject_data(
            subject,
            deriv=opj(base_dir, "derivatives"),
            model=model,
            fix_bold=fix_bold,
            verbose=verbose,
            skip_epi=skip_epi,
            skip_lines=skip_lines,
            overwrite=overwrite
        )

        full_dict[subject] = dd

if __name__ == "__main__":
    main(sys.argv[1:])

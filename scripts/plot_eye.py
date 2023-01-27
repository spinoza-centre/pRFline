#!/usr/bin/env python

import os
import sys
import getopt
from linescanning import (
    dataset, 
    utils,
    plotting)
import numpy as np
import pRFline
from pRFline.utils import SubjectsDict
import matplotlib.pyplot as plt
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """plot_eye.py

Plot the traces of the eyetracking data

Parameters
----------
-s|--subject        process specific subject. Default = "all"


Example
----------
>>> ./plot_eye.py
>>> ./plot_eye.py -s sub-001
    """

    subject = "all"
    verbose = False

    try:
        opts = getopt.getopt(argv,"h:s:v:",["help", "subject=", "verbose"])[0]
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
        elif opt in ("-v", "--verbose"):
            verbose = True            

    utils.verbose("\plot_eye.py", verbose)

    # set defaults
    base_dir    = "/data1/projects/MicroFunc/Jurjen/projects/VE-pRF"
    fig_dir     = opj(opd(opd(pRFline.__file__)), "results")

    # fetch subject dictionary from pRFline.utils.SubjectsDict
    subj_obj = SubjectsDict()
    dict_data = subj_obj.dict_data

    if subject == "all":
        process_list = list(dict_data.keys())
    else:
        process_list = [subject]

    for ix,subject in enumerate(process_list):

        utils.verbose(f"\n**************************************** Processing {subject} ***************************************", verbose)
        
        # read subject-specific session from dictionary
        ses = subj_obj.get_session(subject)

        # find all edf files in sourcedata folder
        log_dir = opj(base_dir, "sourcedata") #, subj, f"ses-{ses}")
        edf_files = utils.FindFiles(log_dir, extension="edf").files
        edf_files

        # get subject-specic 
        sub_edfs = utils.get_file_from_substring(subject, edf_files, exclude="run-0")

        # define eyetracker object
        eye_ = dataset.ParseEyetrackerFile(
            sub_edfs,
            use_bids=True,
            verbose=True,
            h5_file=opj(
                base_dir, 
                "sourcedata", 
                subject, 
                f"ses-{ses}", 
                f"{subject}_ses-{ses}_task-pRF_desc-eyetracker.h5")
        )

        # gaze x/y
        df_gaze = eye_.df_space_eye.copy()

        fig,axs = plt.subplots(
            nrows=len(eye_.edf_file), 
            figsize=(24,len(eye_.edf_file)*6))

        for ix,rr in enumerate(eye_.edf_file):
            
            if ix == len(eye_.edf_file)-1:
                x_lbl = "time (s)"
            else:
                x_lbl = None

            run = utils.split_bids_components(rr)["run"]
            df = utils.select_from_df(df_gaze, expression=f"run = {run}")

            input_l = [df[f"gaze_{i}_int"].values for i in ["x","y"]]
            avg = [float(input_l[i].mean()) for i in range(len(input_l))]
            std = [float(input_l[i].std()) for i in range(len(input_l))]
            
            # x-axis
            xx = list(np.arange(0,input_l[0].shape[0])*1/eye_.sample_rate)
            ax = axs[ix]
            plotting.LazyPlot(
                input_l,
                xx=xx,
                line_width=2,
                axs=ax,
                color=["#1B9E77","#D95F02"],
                labels=[f"gaze {i} (M={round(avg[ix],2)}; SD={round(std[ix],2)}px)" for ix,i in enumerate(["x","y"])],
                x_label=x_lbl,
                y_label="position (pixels)",
                add_hline={"pos": avg},
                title=f"gaze position during run-{run} (raw pixels)"
            )

        fname = opj(fig_dir, subject, f"{subject}_desc-gaze_px")
        for ext in ['png','svg']:
            print(f"Writing {fname}.{ext}")
            fig.savefig(
                f"{fname}.{ext}",
                bbox_inches="tight",
                dpi=300,
                facecolor="white"
            )        

if __name__ == "__main__":
    main(sys.argv[1:])

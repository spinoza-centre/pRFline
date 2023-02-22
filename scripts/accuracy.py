#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V
#$ -q short.q@jupiter
#$ -o ../logs

import os
import sys
import getopt
import numpy as np
import pRFline
import pandas as pd
import math
from alive_progress import alive_bar
from pRFline.utils import SubjectsDict
from linescanning import (
    utils,
    transform
)
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """accuracy.py

Run the registration between ses-1 and ses-2 `n_iter` times to get a distribution of the point estimation. Distance between used coordinate and coordinate found with each iteration of registration is estimated by euclidian distance between two 3D points (see e.g., https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean)

Parameters
----------
-s|--subject        process specific subject. Default = "all"
-i|--n_iters        number of iterations to run (default = 100)

Returns
----------
The registration files of all iterations + <subject>_hemi-L_desc-reg_accuracy.npy file

Example
----------
>>> # fit individual runs of 2D-EPI data & avg of line-scanning data
>>> python accuracy.py
>>> python accuracy -s sub-001
>>> qsub -N s001_acc accuracy.py -s sub-001 --n_iters 200
    """

    subject = "all"
    n_iters = 100
    verbose = True

    try:
        opts = getopt.getopt(argv,"hs:i:",["help", "subject", "n_iters=", "no_verbose"])[0]
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
        elif opt in ("--no_verbose"):
            subject = arg            
        elif opt in ("-i", "--n_iters"):
            n_iters = int(arg)

    # set defaults
    base_dir = "/data1/projects/MicroFunc/Jurjen/projects/VE-pRF"
    data_dir = opj(opd(opd(pRFline.__file__)), "data")
    results_dir = opj(opd(opd(pRFline.__file__)), "results")

    # fetch subject dictionary from pRFline.utils.SubjectsDict
    subj_obj = SubjectsDict()

    if subject == "all":
        process_list = subj_obj.get_subjects()
        check_file_exist = True
    else:
        process_list = [subject]
        check_file_exist = False

    utils.verbose("accuracy.py\n", verbose)

    df_reg = []
    fname_all = opj(data_dir, f"sub-all_desc-registration.csv")
    for subject in process_list:

        # get session
        ses = subj_obj.get_session(subject)

        # get reference and moving image
        mov = opj(
            base_dir, 
            'derivatives', 
            'freesurfer', 
            subject, 
            'mri',
            "orig.nii.gz")

        ref = utils.get_file_from_substring(f"{subject}_ses-{ses}_acq-MP2RAGE_T1w.nii.gz", opj(
            base_dir, 
            subject, 
            f'ses-{ses}', 
            'anat'))

        # get target coordinate
        file_fs = opj(os.path.dirname(mov), f"{subject}_space-fs_hemi-L_vert-{subj_obj.get_target(subject)}_desc-lps.csv")
        file_ses2 = opj(os.path.dirname(mov), f"{subject}_space-ses{ses}_hemi-L_vert-{subj_obj.get_target(subject)}_desc-lps.csv")
        target_ses2 = np.array(utils.read_chicken_csv(file_ses2))

        for ff,tag in zip([mov,ref],["moving","reference"]):
            if isinstance(ff, list):
                raise ValueError(f"Found multiple files for '{tag}': {ff}")

            if not os.path.exists(ff):
                raise FileNotFoundError(f"Could not find {tag}-image: '{ff}'")
        
        fname = opj(results_dir, subject, f"{subject}_ses-{ses}_hemi-L_desc-reg_accuracy.npy")
        if not os.path.exists(opd(fname)):
            os.makedirs(opd(fname, "reg"), exist_ok=True)
            
        if not os.path.exists(fname):
            
            # initialize empty array
            reg_acc = np.full((n_iters), np.nan)

            # loop through iterations
            utils.verbose(f"Start loop with {n_iters} iterations", verbose)
            with alive_bar(n_iters) as bar:

                for ii in range(n_iters):

                    # define temporary files
                    tmp = opj(results_dir, subject, "reg", f"{subject}_ses-{ses}_iter-{ii+1}_desc-")
                    out_csv = opj(results_dir, subject, "reg", "tmp.csv")
                    out_mat = f"{tmp}genaff.mat"

                    # register orig.nii.gz to low-res anatomy from ses-2
                    if not os.path.exists(out_mat):
                        cmd = f"call_antsregistration {ref} {mov} {tmp}"
                        os.system(cmd)
                    
                    # apply new matrix to original LPS point
                    new_coord = np.array(
                        utils.read_chicken_csv(
                            transform.ants_applytopoints(
                                file_fs, 
                                out_csv, 
                                out_mat)
                            )
                        )

                    # get distance with used ses-2 coordinate
                    eucl = math.dist(target_ses2,new_coord)
                    reg_acc[ii] = eucl

                    # remove temporary files
                    os.remove(out_csv)

                    # progress bar
                    bar()

            utils.verbose(f"Writing {fname}", verbose)
            np.save(fname, reg_acc)

        else:
            reg_acc = np.load(fname)

        utils.verbose(f"{subject}: mean deviation = {round(reg_acc.mean(),2)}mm", verbose)

        # create dataframe and append
        tmp_df = pd.DataFrame(reg_acc, columns=["euclidian"])
        tmp_df["subject"] = subject
        df_reg.append(tmp_df)

    # concatenate list of dataframes
    if len(df_reg) > 0:
        df_reg = pd.concat(df_reg)

    if check_file_exist:
        if len(process_list) > 1:
            utils.verbose(f"Writing {fname_all}", verbose)
            df_reg.to_csv(fname_all)
        else:
            utils.verbose(f"Reading {fname_all}", verbose)
            df_reg = pd.read_csv(fname_all)

if __name__ == "__main__":
    main(sys.argv[1:])

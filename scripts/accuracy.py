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

    try:
        opts = getopt.getopt(argv,"hs:i:",["help", "subject", "n_iters="])[0]
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
        elif opt in ("-i", "--n_iters"):
            n_iters = int(arg)

    # set defaults
    ses         = 2
    base_dir    = "/data1/projects/MicroFunc/Jurjen/projects/VE-pRF"
    design_dir  = opj(opd(opd(pRFline.__file__)), "data")

    # fetch subject dictionary from pRFline.utils.SubjectsDict
    subj_obj = SubjectsDict()
    dict_data = subj_obj.dict_data

    if subject == "all":
        process_list = list(dict_data.keys())
    else:
        process_list = [subject]

    print("accuracy.py\n", flush=True)

    for subject in process_list:

        # get reference and moving image
        mov = utils.get_file_from_substring("orig.nii.gz", opj(
            base_dir, 
            'derivatives', 
            'freesurfer', 
            subject, 
            'mri'))
        ref = utils.get_file_from_substring(f"{subject}_ses-{ses}_acq-MP2RAGE_T1w.nii.gz", opj(
            base_dir, 
            subject, 
            f'ses-{ses}', 
            'anat'))

        # get target coordinate
        target_fs = opj(os.path.dirname(mov), f"{subject}_space-fs_hemi-L_vert-{subj_obj.get_target(subject)}_desc-lps.csv")
        target_ses2 = np.array(utils.read_chicken_csv(opj(os.path.dirname(mov), f"{subject}_space-ses{ses}_hemi-L_vert-{subj_obj.get_target(subject)}_desc-lps.csv")))
        
        fname = opj(design_dir, subject, f"{subject}_hemi-L_desc-reg_accuracy.npy")
        if not os.path.exists(fname):

            # loop through iterations
            reg_acc = []
            print(f"Start loop with {n_iters} iterations", flush=True)
            for ii in range(n_iters):

                # define temporary files
                tmp = opj(design_dir, subject, f"tmp{ii+1}_desc-")
                out_csv = opj(design_dir, subject, "tmp.csv")

                out_mat = f"{tmp}genaff.mat"

                # register orig.nii.gz to low-res anatomy from ses-2
                if not os.path.exists(out_mat):
                    os.system(f"call_antsregistration {ref} {mov} {tmp} rigid")
                
                # apply new matrix to original LPS point
                new_coord = np.array(utils.read_chicken_csv(transform.ants_applytopoints(target_fs, out_csv, out_mat)))

                # get distance with used ses-2 coordinate
                dist = np.linalg.norm(target_ses2-new_coord)
                
                reg_acc.append(dist)

                # remove temporary files
                os.remove(out_csv)

            print(f"Writing {fname}", flush=True)
            np.save(fname, np.array(reg_acc))

        else:
            print(f"{fname} already exists", flush=True)

if __name__ == "__main__":
    main(sys.argv[1:])

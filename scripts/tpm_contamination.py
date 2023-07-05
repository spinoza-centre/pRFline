#!/usr/bin/env python

import os
import sys
import getopt
import pRFline
from pRFline.utils import SubjectsDict
from linescanning import utils
import nibabel as nb
import numpy as np
import pandas as pd
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """tpm_contamination.py

Calculate the contamination of CSF/WM in the ribbon voxels

Parameters
----------
    -s|--subject        process specific subject. Default = "all"
    -v|--no_verbose     turn off verbose (best to have verbose on by default)

Returns
----------
    pandas dataframe with the wm/csf contribution ribbon voxels for each subject/run

Example
----------
>>> ./tpm_contamination.py
>>> ./tpm_contamination.py -s sub-001
    """

    verbose = True
    subject = "all"
    overwrite = False

    try:
        opts = getopt.getopt(argv,"hos:v:",["subject=", "no_verbose", "ow", "overwrite"])[0]
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
        elif opt in ("--overwrite", "-o", "--ow"):
            overwrite = True

    utils.verbose("\ntpm_contamination.py", verbose)

    # set defaults
    base_dir    = "/data1/projects/MicroFunc/Jurjen/projects/VE-pRF"
    design_dir  = opj(opd(opd(pRFline.__file__)), "data")
    fig_dir     = opj(opd(opd(pRFline.__file__)), "results")

    # fetch subject dictionary from pRFline.utils.SubjectsDict
    subj_obj = SubjectsDict()
    dict_data = subj_obj.dict_data

    if subject == "all":
        process_list = list(dict_data.keys())
    else:
        process_list = [subject]

    df_cont = []
    data_dir = opj(opd(opd(pRFline.__file__)), "data")
    fn_cont = opj(data_dir, "sub-all_desc-contamination.csv")
    
    if not os.path.exists(fn_cont) or overwrite:
        for ix,subject in enumerate(process_list):

            utils.verbose(f"\n**************************************** Processing {subject} ***************************************", verbose)
            
            # read subject-specific session from dictionary
            ses = subj_obj.get_session(subject)

            # find ribbon file
            src_dir = opj(
                base_dir,
                subject,
                f"ses-{ses}",
                "func"
                )

            image = utils.get_file_from_substring(["run-1", "desc-ribbon", ".nii.gz"], src_dir)
            if not os.path.exists(image):
                raise FileNotFoundError(f"Could not find ribbon file, create it with 'call_createribbon'")
            else:
                rib_voxels = nb.load(image).get_fdata()

            # find cruise files
            cruise_dir = opj(base_dir, "derivatives", "nighres", subject, f"ses-{ses}")
            cruise_files = utils.get_file_from_substring(["run-1", "cruise_cortex"], cruise_dir)
            
            if isinstance(cruise_files, str):
                cruise_files = [cruise_files]
            
            # loop through runs
            subj_df = {}
            for el in ["run","percentage","tissue","code"]:
                subj_df[el] = []

            for ix,crs in enumerate(cruise_files):

                utils.verbose(f" Dealing with {crs}", verbose)
                
                # read runID from file
                bids_comps = utils.split_bids_components(os.path.basename(crs))
                if "run" in list(bids_comps):
                    runID = bids_comps["run"]
                else:
                    runID = ix+1
                
                # read and rotate 
                rib_voxels = np.rot90(nb.load(image).get_fdata().squeeze())
                seg = np.rot90(nb.load(crs).get_fdata().squeeze())

                # get nr of unique voxels in line direction
                bbox = np.where(rib_voxels>0)
                vox_x = np.unique(bbox[1])
                n_vox = vox_x.shape[0]

                # reshape into nominal shape
                rib_seg = seg[rib_voxels.astype(bool)].reshape(16,n_vox)

                # get nr of occurance relative to total
                total = rib_seg.size
                gm = np.count_nonzero(rib_seg==1)
                wm = np.count_nonzero(rib_seg==2)
                csf = np.count_nonzero(rib_seg==0)
                
                # make percentage
                tpm_gm = (gm/total)*100
                tpm_wm = (wm/total)*100
                tpm_csf = (csf/total)*100

                # store in df
                for code,key,val in zip([0,1,2],["gm","wm","csf"],[tpm_gm,tpm_wm,tpm_csf]):
                    subj_df["percentage"].append(val)
                    subj_df["tissue"].append(key)
                    subj_df["code"].append(code)
                    subj_df["run"].append(runID)

            subj_df = pd.DataFrame(subj_df)
            subj_df["subject"] = subject
            df_cont.append(subj_df)

    if len(df_cont) > 1:
        utils.verbose(f"Writing '{fn_cont}'", verbose)
        df_cont = pd.concat(df_cont).to_csv(fn_cont)

if __name__ == "__main__":
    main(sys.argv[1:])

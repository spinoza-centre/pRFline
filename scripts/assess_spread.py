#!/usr/bin/env python
import os
import sys
import getopt
import pRFline
from pRFline.plotting import pRFSpread
from linescanning import (
    utils,
    prf,
    plotting,
    dataset
)
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
opj = os.path.join

def main(argv):

    """assess_spread.py

    Assess the inherent run-to-run spread of pRF parameters in line-scanning & 2D EPI whole brain data. A few settings are hard coded (i.e., gray matter voxels, screen distance, project paths), but which subject is to be processed can be set with the '-s' flag

    Parameters
    ----------
    -s <subject ID>     subject ID as used throughout the pipeline (e.g., 'sub-001')
    --skip_lines        skip the line-scanning fitting; only do 2D-EPI whole brain stuff
    --skip_epi          do not fit 2D-EPI whole brain data

    Returns
    ----------
    A bunch of figures in the `pRFline`-repository ending with `desc-spread_{lines|2depi}.svg`.

    Example
    ----------
    >>> # fetch results of pRF-estimation with HRF
    >>> python assess_spread.py -s sub-002
    """
    
    subject = None
    ses = 2
    task = "task-pRF"
    lines = True
    epi = True

    try:
        opts = getopt.getopt(argv,"hs:",["sub=", "skip_lines", "skip_epi"])[0]
    except getopt.GetoptError:
        print("ERROR while handling arguments.. Did you specify an 'illegal' argument..?")
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts: 
        if opt == '-h':
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--sub"):
            subject = arg
        elif opt in ("--skip_lines"):
            lines = False
        elif opt in ("--skip_epi"):
            epi = False    

    if len(argv) == 0:
        print(main.__doc__)
        sys.exit()

    # set directories
    project_dir = os.environ.get("DIR_PROJECTS")
    base_dir    = opj(project_dir, 'VE-pRF')
    func_dir    = opj(base_dir, subject, f"ses-{ses}", "func")
    anat_dir    = opj(os.path.dirname(func_dir), 'anat')
    module_path = Path(pRFline.__file__)
    fig_dir     = module_path.parents[1]/'results'/subject

    # create output directory if it doesn't exist
    fig_dir.mkdir(parents=True, exist_ok=True)

    # set ribbon voxels for each subject
    dict_data = {
        "sub-001": {
            "ribbon": (356,364),
            "exclude": "run-1",
            "target": 2265,
            "screen_size": 70
        },
        "sub-002": {
            "ribbon": (355,363),
            "exclude": None,
            "target": 2249,
            "screen_size": 39.3
        },
        "sub-003": {
            "ribbon": (356,365),
            "exclude": "run-4",
            "target": 646,
            "screen_size": 39.3
        },
        "sub-007": {
            "ribbon": (361,367),
            "exclude": None,
            "target": 4578,
            "screen_size": 39.3
        },
        "sub-008": {
            "ribbon": (358,364),
            "exclude": "run-2",
            "target": 10009,
            "screen_size": 39.3
        }
    }

    if lines:

        run_files   = utils.get_file_from_substring(
            [subject, f"ses-{ses}", f"{task}"], 
            func_dir, 
            exclude=dict_data[subject]["exclude"])
            
        func_file = utils.get_file_from_substring("bold.mat", run_files)
        anat_slices = utils.get_file_from_substring([subject, f"ses-{ses}", "acq-1slice", ".nii.gz"], anat_dir)
        ref_slices  = utils.match_lists_on(func_file, anat_slices, matcher='run')

        # mind you, the segmentations live in ses-1 space, NOT FREESURFER!
        ses_to_motion = utils.get_file_from_substring(f"ses{ses}_rec-motion1", opj(base_dir, "derivatives", 'pycortex', subject, 'transforms'))
        run2run = utils.get_file_from_substring(['.txt'], anat_dir)

        #---------------------------------------------------------------------------------------------------
        # PREPROCESSING

        data_obj = dataset.Dataset(
            func_file,
            use_bids=True,
            verbose=True,
            acompcor=True,
            ref_slice=ref_slices,
            ses1_2_ls=ses_to_motion,
            run_2_run=run2run,
            n_pca=5,
            deleted_last_timepoints=300,
            report=False)

        df_func = data_obj.fetch_fmri()

        # select ribbon voxels
        df_ribbon = utils.select_from_df(df_func, expression="ribbon", indices=dict_data[subject]["ribbon"])

        # read design matrix
        dm_f = opj(str(fig_dir), f"{subject}_desc-full_design.mat")
        if os.path.exists(dm_f):
            print(f"Reading {dm_f}")
            dm_ = prf.read_par_file(dm_f)

        # read run-specific ribbon data
        run_ids = data_obj.get_runs(df_ribbon)
        run_data = [utils.select_from_df(df_ribbon, expression=f"run = {ii}").values.mean(axis=-1) for ii in run_ids]

        # sync design matrix and data shapes
        sync_dm = dm_[...,:run_data[0].shape[0]]

        #---------------------------------------------------------------------------------------------------
        # DESCRIPTIVES; design matrix on run-specific averages

        tc_f = opj(str(fig_dir), f"{subject}_desc-avg_ribbon_runs.svg")
        fig,axs = plt.subplots(figsize=(24,6))

        # add shaded area where stim was on screen
        for ii in range(sync_dm.shape[-1]):
            if not np.all(sync_dm[...,ii] == 0):
                axs.axvspan(ii, ii+1, alpha=0.2, color="#cccccc")

        plotting.LazyPlot(
            run_data,
            axs=axs,
            line_width=1,
            add_hline=0,
            labels=[f"run-{ii}" for ii in run_ids],
            x_lim=[0,run_data[0].shape[0]+200],
            x_ticks=list(np.arange(0,run_data[0].shape[0]+200,400)),
            x_label="volumes",
            y_label="magnitude (%change)",
            title=f"{subject}: average across ribbon {dict_data[subject]['ribbon']} [colors denote separate runs]",
        )

        fig.savefig(tc_f, dpi=300, bbox_inches='tight', facecolor="white")    

        #---------------------------------------------------------------------------------------------------
        # LINESCANNING FITTING

        fits_line = {}
        for ii in range(len(run_ids)):

            print("\n---------------------------------------------------------------------------------------------------")
            print(f"Dealing with run-{run_ids[ii]}")

            data = run_data[ii][np.newaxis,:dm_.shape[-1]]
            obj_ = prf.pRFmodelFitting(
                data,
                design_matrix=sync_dm,
                model="gauss",
                stage="iter",
                TR=0.105,
                fix_bold_baseline=True,
                verbose=True,
                rsq_threshold=0,
                screen_distance_cm=196
            )

            obj_.fit()
            fits_line[ii] = obj_

        # print r2 for each run
        for ix,key in enumerate(list(fits_line.keys())):
            if key != "avg":
                print(f"run-{key+1}; r2={round(fits_line[key].gauss_iter[0,-1],2)}")

        # average runs
        avg_psc = df_ribbon.groupby(['subject', "t"]).mean().values.mean(axis=-1)[np.newaxis,:]

        # fit average
        avg_ = prf.pRFmodelFitting(
            avg_psc,
            design_matrix=sync_dm,
            model="gauss",
            stage="iter",
            TR=0.105,
            fix_bold_baseline=True,
            verbose=True,
            rsq_threshold=0,
            screen_distance_cm=196,
            grid_nr=40
        )

        avg_.fit()
        fits_line['avg'] = avg_

        # plot spread
        fig_lines = pRFSpread(
            fits_line,
            subject=subject,
            model="gauss",
            fig_dir=fig_dir,
            pad_title=20
        )

        fig_lines.plot_spread()

        # plot fits individual runs
        fig_lines.plot_fits()

        # save files
        fig_lines.save_fig()

    #---------------------------------------------------------------------------------------------------
    # 2D EPI WHOLE BRAIN FITTING        

    if epi:
        data_dir = opj(base_dir, "derivatives", "pybest", subject, "ses-1", "unzscored")
        data_files = utils.get_file_from_substring("npy", data_dir, exclude="hemi-R")
        design = utils.resample2d(prf.read_par_file(opj(base_dir, "derivatives", "prf", subject, "ses-1", "design_task-2R.mat")), new_size=100)

        # remove first 4 volumes
        cut_vols = 4
        design_cut = design.copy()[...,cut_vols:]

        collect_vox = []
        fits_epi = {}
        for ix,data in enumerate(data_files):
            
            print("\n---------------------------------------------------------------------------------------------------")
            print(f"Dealing with run-{ix+1}")    
            
            # get target vertex data
            vox_data = np.load(data)[:,dict_data[subject]["target"]][np.newaxis,cut_vols:]

            # convert to percent change
            vox_psc = utils.percent_change(vox_data, 1, baseline=15)

            # append
            collect_vox.append(vox_psc)

            # fit
            fit_ = prf.pRFmodelFitting(
                vox_psc,
                design_matrix=design_cut,
                fix_bold_baseline=True,
                model="gauss",
                verbose=True,
                rsq_threshold=0,
                TR=1.5,
                screen_distance_cm=210,
                screen_size_cm=dict_data[subject]["screen_size"]
            )

            fit_.fit()
            fits_epi[ix] = fit_

        # average
        collect_vox = np.concatenate(collect_vox, axis=0)
        avg = np.median(collect_vox, axis=0, keepdims=True)

        for ix,key in enumerate(list(fits_epi.keys())):
            if key != "avg":
                print(f"run-{key+1}; r2={round(fits_epi[key].gauss_iter[0,-1],2)}\t{os.path.basename(data_files[ix])}")

        # fit average
        avg_ = prf.pRFmodelFitting(
            avg,
            design_matrix=design_cut,
            fix_bold_baseline=True,
            model="gauss",
            verbose=True,
            rsq_threshold=0,
            TR=1.5,
            screen_distance_cm=210,
            screen_size_cm=dict_data[subject]["screen_size"]
        )

        avg_.fit()
        fits_epi['avg'] = avg_

        # initiate plot object
        fig_epi = pRFSpread(
            fits_epi,
            subject=subject,
            model="gauss",
            fig_dir=fig_dir,
            data_type="2depi"
        )

        # spread plot
        fig_epi.plot_spread()    

        # fit plot
        fig_epi.plot_fits()

        # save
        fig_epi.save_fig()

    print("Done")

if __name__ == "__main__":
    main(sys.argv[1:])

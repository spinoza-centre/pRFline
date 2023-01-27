import os
import numpy as np
from linescanning import prf,utils
from datetime import datetime
opj = os.path.join

class SubjectsDict():

    def __init__(self):

        # set ribbon voxels for each subject
        self.dict_data = {
            "sub-001": {
                # "ribbon": (356,364),
                "ribbon": (359,366),
                "exclude": [1],
                "target": 1053,
                "screen_size": 70,
                "line_ses": 2,
                "invert": False,
                "bounds": True,
                "views": {
                    "azimuth": 180,
                    "altitude": 105,
                    "radius": 163
                },
            },
            "sub-002": {
                "ribbon": (358,366),
                "exclude": [],
                "target": 2249,
                "screen_size": 39.3,
                "line_ses": 2,
                "invert": False,
                "bounds": True,
                "views": {
                    "azimuth": 180,
                    "altitude": 105,
                    "radius": 163
                },
            },
            "sub-003": {
                "ribbon": (358,366),
                "exclude": [4],
                "target": 646,
                "screen_size": 70,
                "line_ses": 2,
                "invert": False,
                "bounds": False,
                "views": {
                    "azimuth": 180,
                    "altitude": 105,
                    "radius": 163
                },
            },
            "sub-005": {
                "ribbon": (360,366),
                "exclude": [],
                "target": 6055,
                "screen_size": 39.3,
                "line_ses": 3,
                "invert": False,
                "bounds": False,
                "views": {
                    "azimuth": 180,
                    "altitude": 105,
                    "radius": 163
                },
            },
            "sub-007": {
                "ribbon": (361,366),
                "exclude": [1],
                "target": 4578,
                "screen_size": 39.3,
                "line_ses": 2,
                "invert": False,
                "bounds": True,
                "views": {
                    "azimuth": 180,
                    "altitude": 105,
                    "radius": 163
                },
            },
            "sub-008": {
                "ribbon": (359,364),
                "exclude": [2],
                "target": 10009,
                "screen_size": 39.3,
                "line_ses": 2,
                "invert": False,
                "bounds": True,
                "views": {
                    "azimuth": 180,
                    "altitude": 105,
                    "radius": 163
                },
            },
            # "sub-009": {
            #     "ribbon": (358,367),
            #     "exclude": [2],
            #     "target": 1298,
            #     "screen_size": 39.3,
            #     "line_ses": 2,
            #     "invert": True,
            #     "bounds": True
            # }            
        }

    def get_subjects(self):
        return list(self.dict_data.keys())

    def get_views(self, subject):
        return self.dict_data[subject]["views"]

    def get_target(self, subject):
        return self.dict_data[subject]["target"]
    
    def get_ribbon(self, subject):
        return self.dict_data[subject]["ribbon"]

    def get_exclude(self, subject):
        return self.dict_data[subject]["exclude"]

    def get_screen_size(self, subject):
        return self.dict_data[subject]["screen_size"]

    def get_session(self, subject):
        return self.dict_data[subject]["line_ses"]

    def get_invert(self, subject):
        return self.dict_data[subject]["invert"]

    def get_bounds(self, subject):
        return self.dict_data[subject]["bounds"]        

def sort_posthoc(df):

    conditions = np.unique(np.array(list(df["A"].values)+list(df["B"].values)))

    distances = []
    for contr in range(df.shape[0]): 
        A = df["A"].iloc[contr]
        B = df["B"].iloc[contr]

        x1 = np.where(conditions == A)[0][0]
        x2 = np.where(conditions == B)[0][0]

        distances.append(abs(x2-x1))
    
    df["distances"] = distances
    return df.sort_values("distances", ascending=False)

def read_subject_data(
    subject,
    deriv=None,
    model="gauss",
    fix_bold=True,
    verbose=False,
    skip_lines=False,
    skip_epi=False,
    overwrite=False):

    subj_obj = SubjectsDict()

    if not isinstance(deriv, str):
        deriv = os.environ.get("DIR_DATA_DERIV")

    # get subject-specific session ID; sub-005 has multiple sessions, of which ses-2 was bad
    ses = subj_obj.get_session(subject)

    # define output directory
    prf_dir = opj(
        deriv, 
        "prf",
        subject, 
        f"ses-{ses}")
    
    # initiate dictionary
    subject_dict = {}

    utils.verbose("\n---------------------------------------------------------------------------------------------------", verbose)
    utils.verbose(f"Dealing with {subject} [{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}]", verbose)

    if not skip_lines:
        utils.verbose("\n-- linescanning --", verbose)
        
        # add line-scanning key
        subject_dict["lines"] = {}

        ####################################################################################################
        # LINE-SCANNING BIT
        ####################################################################################################

        #---------------------------------------------------------------------------------------------------
        # get low-res design 
        dm_f = opj(prf_dir, f"{subject}_ses-{ses}_task-pRF_run-avg_desc-design_matrix.mat")
        if os.path.exists(dm_f):
            utils.verbose(f"Reading {dm_f}", verbose)
            dm_ = prf.read_par_file(dm_f) 
        else:
            raise FileNotFoundError(f"Could not find file '{dm_f}'")

        for tt in ["all","avg","ribbon"]:

            input_data = prf.read_par_file(opj(os.path.dirname(dm_f), f"{subject}_ses-{ses}_task-pRF_run-avg_vox-{tt}_desc-data.npy"))
            obj_ = prf.pRFmodelFitting(
                input_data.T,
                design_matrix=dm_,
                model=model,
                TR=0.105,
                fix_bold_baseline=fix_bold,
                verbose=verbose,
                rsq_threshold=0,
                screen_distance_cm=196,
                grid_nr=40,
                write_files=True,
                save_grid=False,
                output_dir=prf_dir,
                output_base=f"{subject}_ses-{ses}_task-pRF_run-avg_vox-{tt}"
            )

            pars_file = opj(obj_.output_dir, f"{obj_.output_base}_model-{model}_stage-iter_desc-prf_params.pkl")
            if not os.path.exists(pars_file) or overwrite:
                utils.verbose(f"Could not find file '{pars_file}'. Run 'python line_fit.py with --{model}'", verbose)
            else:
                utils.verbose(f"Reading {pars_file}", verbose)
                obj_.load_params(
                    pars_file, 
                    model=model, 
                    stage="iter")

            subject_dict["lines"][tt] = obj_

        utils.verbose("Done", verbose)

    ####################################################################################################
    # 2D-EPI WHOLE-BRAIN BIT
    ####################################################################################################

    if not skip_epi:
        utils.verbose("\n-- whole brain 2D-EPI --", verbose)
        
        # add whole-brain key
        subject_dict["wb"] = {}

        # get directories
        data_dir = opj(
            deriv, 
            "pybest",
            subject, 
            "ses-1",
            "unzscored")

        data_files = utils.get_file_from_substring(["hemi-L","npy"], data_dir)

        design_fn = opj(os.path.dirname(prf_dir), "ses-1", "design_task-2R.mat")
        design = utils.resample2d(prf.read_par_file(design_fn), new_size=100)

        # remove first 4 volumes
        cut_vols = 4
        design_cut = design.copy()[...,cut_vols:]

        collect_vox = []
        utils.verbose(f"Reading functional data from '{data_dir}'", verbose)
        for _,data in enumerate(data_files):
            
            # get target vertex data
            vox_data = np.load(data)[cut_vols:,:]

            # convert to percent change
            vox_psc = utils.percent_change(vox_data, 0, baseline=15)

            # append
            collect_vox.append(vox_psc[...,np.newaxis])

        # concatenate <time,vertices,runs>
        collect_vox = np.concatenate(collect_vox, axis=-1)

        # dont have 2DEPI fits for CSS/DoG model, default to Gauss unless model==norm
        if model == "norm":
            load_md = model
        else:
            load_md = "gauss"

        # fit average
        runs_epi = prf.pRFmodelFitting(
            collect_vox[:,subj_obj.get_target(subject),:].T,
            design_matrix=design_cut,
            fix_bold_baseline=fix_bold,
            model=load_md,
            verbose=verbose,
            rsq_threshold=0.05,
            TR=1.5,
            screen_distance_cm=210,
            screen_size_cm=subj_obj.get_screen_size(subject),
            write_files=True,
            save_grid=False,
            output_dir=os.path.dirname(design_fn),
            output_base=f"{subject}_acq-2DEPI_run-all",
            nr_jobs=1
        )


        pars_file = opj(runs_epi.output_dir, f"{runs_epi.output_base}_model-{load_md}_stage-iter_desc-prf_params.pkl")
        if not os.path.exists(pars_file) or overwrite:
            utils.verbose(f"Fitting all runs [data={runs_epi.data.shape}]..", verbose)
            runs_epi.fit()
        else:
            utils.verbose(f"Reading {pars_file}", verbose)
            runs_epi.load_params(
                pars_file, 
                model=load_md, 
                stage="iter")

        subject_dict["wb"]['runs'] = runs_epi

        # collect avg whole brain pRF estimates
        avg = np.median(collect_vox, axis=-1)
        avg_epi = prf.pRFmodelFitting(
            avg.T,
            design_matrix=design_cut,
            model=load_md,
            verbose=verbose,
            TR=1.5,
            screen_distance_cm=210
        )

        pars_file = opj(os.path.dirname(design_fn), f"{subject}_ses-1_task-2R_roi-V1_model-{load_md}_stage-iter_desc-prf_params.pkl")
        
        if not os.path.exists(pars_file):
            raise FileNotFoundError(f"Could not find file '{pars_file}'. Run 'master -m 17' with '--v1' and '--norm'")
        else:
            utils.verbose(f"Reading {pars_file}", verbose)
            avg_epi.load_params(
                pars_file, 
                model=load_md, 
                stage="iter")

        subject_dict["wb"]['avg'] = avg_epi
        subject_dict["subject"] = subject

    return subject_dict

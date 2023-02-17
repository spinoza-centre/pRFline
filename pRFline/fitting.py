from linescanning import (
    utils, 
    prf, 
    dataset, 
    plotting)
from linescanning.fitting import CurveFitter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D         
import numpy as np
import os
from scipy import io
import seaborn as sns

opj = os.path.join

class FitpRFs(dataset.Dataset):
    """FitpRFs

    Fitting object for for the `pRFline`-experiment, in which we aim to reidentify the target pRF using 3D-EPIs and line-scanning data. This class creates the design matrix for the run given a log-directory, applies low/high pass filtering, and fits the pRFs to the data. In case of line-scanning data, we average runs and iterations. Partial FOV data should be preprocessed following the instructions here: https://github.com/spinoza-centre/pRFline/tree/main/data

    Parameters
    ----------
    func_files: str, optional
        List of input files
    TR: float, optional
        Repetition time of partial FOV acquisition, by default 1.111
    log_dir: str, optional
        Path-like representation pointing to the log-directory in which a directory with `Screenshots` lives, by default None. See https://github.com/gjheij/linescanning/blob/main/linescanning/prf.py#L756 for more details on the creation of the design matrix
    output_base: str, optional
        Basename for output, by default None. If none, it will try to derive a base name using BIDS-components from `func_files`. If this is not possible, an error will be thrown with a request to specify a output basename or format your input files according to BIDS
    output_dir: str, optional
        Path to the output directory, by default None. If None, the output will be stored in the directory in which `func_files` lives
    verbose: bool, optional
        Turn on messages, by default False
    strip_baseline: bool, optional
        Strip the design matrix from its baseline, default=True
    ribbon: tuple, list, optional
        If a list or tuple of voxels representing the gray-matter ribbon is specified, we'll perform the pRF-fit only on these voxels. This is generally favourable, but because the ribbon differs across subjects the default is None.
    gm_only: bool, optional
        Fit on all gray matter voxels; mutually exclusive wih `ribbon`-argument
    average: bool, optional
        Average across runs before throwing data into the fitter. Default = True. Allows you to fit data on separate runs as well
    rsq_threshold: float, optional
        Manual set threshold for r2 if the one specified in the `template` is not desired.
    fit_hrf: bool, optional
        Fit the HRF during pRF-fitting. If `True`, the fitting will consist of two stages: first, a regular fitting without HRF estimatiobn. Then, the fitting object of that fit is inserted as `previous_gaussian_fitter` into a new fitter object with HRF estimation turned on. Default = False.
    is_lines: bool, optional
        There's a slightly different procedure for line-scanning compared to whole-brain or 3D-EPI: for the line-scanning, we have two iterations per run. So we need to average those in order to get a single iteration. By default True
    no_grid: bool, optional
        Don't save grid-parameters; can save clogging up of directories. Default = False

    Raises
    ----------
    ValueError
        If no output basename was specified, and the input file is not formatted according to BIDS
    ValueError
        If the **first** dimension of the functional data does not match the **last** dimension of the design matrix (time)

    Example
    ----------
    >>> func_files = get_file_from_substring(["task-pRF", "bold.npy"], func_dir)
    >>> model_fit = fitting.FitpRFs(
    >>>     func_files=func_files,
    >>>     output_dir=output_dir,
    >>>     TR=0.105,
    >>>     log_dir=log_dir,
    >>>     model='norm',
    >>>     verbose=True)
    >>> model_fit.fit()

    Notes
    ----------
    Because we also need to filter our data in this class, we can specify any argument accepted by :class:`linescanning.dataset.Dataset` via `kwargs`.
    """

    def __init__(
        self, 
        func_files, 
        TR=0.105, 
        log_dir=None, 
        output_base=None, 
        output_dir=None, 
        verbose=False, 
        baseline_duration=30, 
        iter_duration=160, 
        n_iterations=3,
        strip_baseline=True, 
        ribbon=None,
        gm_only=False,
        average=True,
        rsq_threshold=None,
        fit_hrf=False,
        fix_parameters=None,
        n_pix=100,
        design_only=False,
        is_lines=True,
        start_avg=False,
        save_grid=True,
        overwrite=False,
        full_design=False,
        **kwargs):

        self.func_files         = func_files
        self.verbose            = False
        self.TR                 = TR
        self.log_dir            = log_dir
        self.duration           = 0.25
        self.output_dir         = output_dir
        self.model              = "norm"
        self.stage              = "iter"
        self.verbose            = verbose
        self.baseline_duration  = baseline_duration
        self.iter_duration      = iter_duration
        self.n_iterations       = n_iterations
        self.strip_baseline     = strip_baseline
        self.ribbon             = ribbon
        self.average            = average
        self.rsq_threshold      = rsq_threshold
        self.fit_hrf            = fit_hrf
        self.n_pix              = n_pix
        self.design_only        = design_only
        self.is_lines           = is_lines
        self.gm_only            = gm_only
        self.fix_parameters     = fix_parameters
        self.start_avg          = start_avg
        self.save_grid          = save_grid
        self.overwrite          = overwrite
        self.full_design        = full_design

        # try to derive output name from BIDS-components in input file
        if output_base == None:
            try:
                if isinstance(func_files, list):
                    func = func_files[0]
                else:
                    func = func_files

                comps = utils.split_bids_components(func)
                if self.is_lines:
                    acq = "line"
                else:
                    acq = "3DEPI"

                self.output_base = f"sub-{comps['sub']}_ses-{comps['ses']}_task-{comps['task']}_acq-{acq}"
            except:
                raise ValueError(f"Could not read BIDS-components from {func}. Please format accordingly or specify 'output_base'")
        else:
            self.output_base = output_base

        # get bids components
        self.bids_comps = utils.split_bids_components(self.output_base)
        
        # create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # update kwargs for pRFmodelFitting & plotting
        self.__dict__.update(kwargs)

        if self.log_dir == None:
            raise ValueError(f"Please specify a log-directory..")
        
        # fetch data
        self.prepare_func(**kwargs)

        # average iterations or assume experiment is 1 iteration
        if self.is_lines:
            self.average_iterations(**kwargs)
        else:
            if hasattr(self, "avg"):
                self.avg_iters_baseline = self.avg.copy()
            else:
                raise ValueError("Could not find 'avg'-element. Did you run 'prepare_func()' with 'is_lines=False'?")
        
        # check if we should make design; default = False, which results in the DM being created in fit()
        if self.design_only:
            self.prepare_design()

    def fit(self, **kwargs):

        counter = 1
        if self.start_avg:
            txt = "average across GM"
        else:
            txt = "fitting complete line"

        utils.verbose("\n---------------------------------------------------------------------------------------------------", self.verbose)
        utils.verbose(f"STAGE {counter} [{txt}]", self.verbose)

        # use data with/without baseline
        if self.strip_baseline:
            utils.verbose("Using data WITHOUT baseline for fitting", self.verbose)
            self.data_for_fitter = self.avg_iters_no_baseline.copy()
        else:
            utils.verbose("Using data WITH baseline for fitting", self.verbose)
            self.data_for_fitter = self.avg_iters_baseline.copy()

        # this option tells the fitter to fit on the average across depth
        fix_pars = None
        if self.start_avg:
            self.stage1_data = self.data_for_fitter.mean(axis=-1)
            append_txt = "_vox-avg"
        else:
            if isinstance(self.ribbon, tuple):
                append_txt = "_vox-ribbon"
            else:
                append_txt = "_vox-all"

            self.stage1_data = self.data_for_fitter.copy()
            
            if isinstance(self.fix_parameters, list):
                fix_pars = self.fix_parameters.copy()
        
        if not hasattr(self, "design"):
            self.prepare_design(**kwargs)
        
        # stage 1 - no HRF
        if hasattr(self, "constraints"):
            constr = self.constraints
        else:
            constr = "tc"

        if hasattr(self, "use_grid_bounds"):
            use_bounds = self.use_grid_bounds
        else:
            use_bounds = True    

        self.stage1 = prf.pRFmodelFitting(
            self.stage1_data.T,
            design_matrix=self.design, 
            TR=self.TR, 
            model=self.model, 
            stage=self.stage, 
            verbose=self.verbose,
            output_dir=self.output_dir,
            output_base=self.output_base+append_txt,
            write_files=True,
            rsq_threshold=self.rsq_threshold,
            fix_bold_baseline=True,
            fix_parameters=fix_pars,
            save_grid=self.save_grid,
            screen_distance_cm=196,
            grid_nr=40,
            constraints=constr,
            fit_hrf=self.fit_hrf,
            use_grid_bounds=use_bounds,
            **kwargs)

        # save data
        data_fname = opj(self.output_dir, self.stage1.output_base+f'_desc-data.npy')
        utils.verbose(f"Saving data as {data_fname}", self.verbose)
        np.save(data_fname, self.stage1_data)

        # check if params-file already exists
        self.pars_file = opj(self.output_dir, self.stage1.output_base+f"_model-{self.model}_stage-{self.stage}_desc-prf_params.pkl")

        # run fit if file doesn't exist, otherwise load them in case we want to fit the prf
        if not os.path.exists(self.pars_file) or self.overwrite:
            utils.verbose(f"Running fit with {self.model}-model ({txt})", self.verbose)

            # inject existing gaussian pars if model != Gauss
            if self.model != "gauss":
                self.gauss_file = opj(self.output_dir, self.stage1.output_base+f"_model-gauss_stage-{self.stage}_desc-prf_params.pkl")
                if os.path.exists(self.gauss_file):
                    utils.verbose(f"Gaussian estimates: '{self.gauss_file}'", self.verbose)
                    self.stage1.old_params = self.gauss_file

            # fit
            self.stage1.fit()
        else:
            # load parameters
            self.pars_file = opj(self.output_dir, self.stage1.output_base+f"_model-{self.model}_stage-{self.stage}_desc-prf_params.pkl")
    
            utils.verbose(f"Parameter file '{self.pars_file}'", self.verbose)
            self.stage1.load_params(
                self.pars_file, 
                model=self.model, 
                stage=self.stage)

        # if start_avg, feed average fits as starting point for ribbon
        if self.start_avg:
            counter += 1
            print("\n---------------------------------------------------------------------------------------------------")
            print(f"STAGE {counter} [ribbon]")

            # base ribbon fit on average fit
            insert_params = getattr(self.stage1, f"{self.model}_iter")

            # Gauss iterfit with pRF-estimates from average ribbon as starting point
            self.stage2_data = self.data_for_fitter.copy()
            self.stage2 = prf.pRFmodelFitting(
                self.stage2_data.T,
                design_matrix=self.design, 
                TR=self.TR, 
                model=self.model, 
                stage=self.stage, 
                verbose=self.verbose,
                output_dir=self.output_dir,
                output_base=self.output_base+"_vox-ribbon",
                write_files=True,
                rsq_threshold=self.rsq_threshold,
                fix_bold_baseline=True,
                fix_parameters=self.fix_parameters,
                old_params=insert_params,
                save_grid=self.save_grid,
                skip_grid=True,
                fit_hrf=self.fit_hrf,
                constraints=constr,
                **kwargs)

            # save data
            data_fname = opj(self.output_dir, self.stage2.output_base+f'_desc-data.npy')
            utils.verbose(f"Saving data as {data_fname}", self.verbose)
            np.save(data_fname, self.stage2_data)

            # check if params-file already exists
            self.pars_file2 = opj(self.output_dir, self.stage2.output_base+f"_model-{self.model}_stage-{self.stage}_desc-prf_params.pkl")
            # run fit if file doesn't exist, otherwise load them in case we want to fit the prf
            if not os.path.exists(self.pars_file2) or self.overwrite:
                utils.verbose(f"Running fit with {self.model}-model (ribbon)", self.verbose)

                # fit
                self.stage2.fit()
            else:
                utils.verbose(f"Parameter file '{self.pars_file2}'", self.verbose)
                
                # load
                self.stage2.load_params(
                    self.pars_file2, 
                    model=self.model, 
                    stage=self.stage)

    def prepare_design(self, **kwargs):
        
        dm_fname = opj(self.output_dir, self.output_base+'_desc-design_matrix.mat')
        if os.path.exists(dm_fname):
            utils.verbose(f"Using existing design matrix: {dm_fname}", self.verbose)
            
            self.design = prf.read_par_file(dm_fname)
        else:
            utils.verbose(f"Using {self.log_dir} for design", self.verbose)

            self.design = prf.create_line_prf_matrix(
                self.log_dir, 
                stim_duration=0.25,
                stim_at_half_TR=True,
                nr_trs=self.avg_iters_baseline.shape[0],
                TR=0.105,
                verbose=self.verbose,
                n_pix=self.n_pix,
                **kwargs)

            # strip design from its baseline
            if self.strip_baseline:
                self.design = self.design[...,self.baseline.shape[0]:]

            # save design matrix for later reference
            utils.verbose(f"Saving design matrix as {dm_fname}", self.verbose)
            io.savemat(dm_fname, {"stim": self.design})

        # check the shapes of design and functional data match
        utils.verbose(f"Design matrix has shape: {self.design.shape}", self.verbose)

        if hasattr(self, "data_for_fitter"):
            if self.verbose:
                if self.design.shape[-1] == self.data_for_fitter.shape[0]:
                    utils.verbose("Shapes of design matrix and functional data match. Ready for fit!", self.verbose)
                else:
                    raise ValueError(f"Shapes of functional data ({self.data_for_fitter.shape[0]}) does not match shape of design matrix ({self.design.shape[-1]}). You might have to transpose your data or cut away baseline.")

    def prepare_func(self, **kwargs):

        if self.verbose:
            print("\n---------------------------------------------------------------------------------------------------")
            print(f"Preprocessing functional data")                    
        
        self.h5_bids = ""
        self.subject = ""
        for ii in ["sub","ses","task"]:
            if len(self.bids_comps.keys()) > 0:
                if ii in list(self.bids_comps.keys()):
                    if ii == "sub":
                        self.subject = f"{ii}-{self.bids_comps[ii]}"

                    self.h5_bids += f"{ii}-{self.bids_comps[ii]}_"
        
        if len(self.h5_bids) == 0:
            self.h5_bids = f"{self.output_base}_desc-preproc_bold.h5"
        else:
            self.h5_bids += "desc-preproc_bold.h5"

        self.h5_file = opj(os.environ.get('DIR_DATA_DERIV'), "lsprep", self.subject, self.h5_bids)
        if os.path.exists(self.h5_file):
            self.func_input = self.h5_file
            self.write_h5 = False
        else:
            self.write_h5 = True
            self.func_input = self.func_files

        super().__init__(
            self.func_input, 
            verbose=self.verbose, 
            **kwargs)

        # save h5 if needed
        if self.write_h5:
            self.to_hdf(output_file=self.h5_file)

        # fetch data and filter out NaNs
        self.data = self.fetch_fmri()

        if isinstance(self.ribbon, tuple):
            self.ribbon = list(np.arange(*list(self.ribbon)))
    
        if isinstance(self.ribbon, list) and not self.gm_only:
            utils.verbose(f" Including voxels: {self.ribbon}", self.verbose)

            self.data = utils.select_from_df(
                self.data,
                expression="ribbon",
                indices=self.ribbon
            )
        
        if self.gm_only:
            self.data = self.gm_df.copy()

        # average
        self.avg = self.data.groupby(['subject', 't']).mean()

    def average_iterations(self, **kwargs):

        # initialize Dataset-parent
        if not hasattr(self, 'avg'):
            self.prepare_func(**kwargs)

        # check if we should remove initial volumes (no advisable)
        if not hasattr(self, "deleted_first_timepoints"):
            start = int(round(self.baseline_duration/self.TR, 0))
        else:
            start = int(round(self.baseline_duration/self.TR, 0)) - int(round(self.deleted_first_timepoints*self.TR, 0))

        # fetch baseline volumes
        self.baseline = self.avg[:start]
        iter_size     = int(round(self.iter_duration/self.TR, 0))
        
        utils.verbose(f" Baseline    = {start} vols (~{round(start*self.TR,2)}s) based on TR of {self.TR}s ({self.baseline_duration}s was specified/requested)", self.verbose)
        utils.verbose(f" 1 iteration = {iter_size} vols (~{round(iter_size*self.TR,2)}s) based on TR of {self.TR}s ({self.iter_duration}s was specified/requested)", self.verbose)

        if self.verbose:
            if self.n_iterations > 1:
                utils.verbose(f" Chunking/averaging {self.n_iterations} iterations", self.verbose)
            elif self.n_iterations == 1:
                utils.verbose(f" Averaging of chunks turned OFF (nr of iterations = {self.n_iterations})", self.verbose)
            else:
                raise ValueError(f"Unknown recognized number of iterations: '{self.n_iterations}'")

        if not self.full_design:
            self.iter_chunks = []
            for ii in range(self.n_iterations):

                # try to fetch values, if steps are out of bounds, zero-pad the timecourse
                if start+iter_size < self.avg.shape[0]:
                    chunk = self.avg.values[start:start+iter_size]
                else:
                    chunk = self.avg.values[start+iter_size:]
                    padded_array = np.zeros((iter_size, self.avg.shape[-1]))
                    padded_array[:chunk.shape[0]] = chunk
                    chunk = padded_array.copy()

                self.iter_chunks.append(chunk[...,np.newaxis])
                start += iter_size

            self.avg_iters_baseline = np.concatenate((self.baseline, np.concatenate(self.iter_chunks, axis=-1).mean(axis=-1)))
            self.avg_iters_no_baseline = np.concatenate(self.iter_chunks, axis=-1).mean(axis=-1)
        else:
            self.avg_iters_baseline = self.avg.values
            self.avg_iters_no_baseline = self.avg_iters_baseline[...,self.baseline.shape[0]:]

        utils.verbose(f" With baseline: {self.avg_iters_baseline.shape} | No baseline: {self.avg_iters_no_baseline.shape}", self.verbose)
from linescanning import utils, prf, dataset
from pRFline.utils import split_params_file
import os
import numpy as np
from scipy import io

opj = os.path.join

class FitPartialFOV:
    """FitPartialFOV

    Fitting object for partial FOV data that has been preprocessed with https://github.com/spinoza-centre/pRFline/blob/main/scripts/partial_preprocess.py. This workflow results in a numpy-file ending with `hemi-LR_space-fsnative_bold.npy`, which is a format compatible with https://github.com/gjheij/linescanning/blob/main/linescanning/dataset.py#L1170. This class creates the design matrix for the run given a log-directory, applies a high-pass filter to remove low-frequency drifts, and performs the fitting

    Parameters
    ----------
    func_files: str, optional
        String representing the functional file that was preprocessed by `partial_preprocessing.py`, by default None
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

    Raises
    ----------
    ValueError
        If no output basename was specified, and the input file is not formatted according to BIDS
    ValueError
        If the **first** dimension of the functional data does not match the **last** dimension of the design matrix (time)

    Example
    ----------
    >>> func_files = get_file_from_substring(["hemi-LR", "bold.npy"], func_dir)
    >>> model_fit = fitting.FitPartialFOV(func_files=func_files,
    >>>                                   output_dir=output_dir,
    >>>                                   TR=1.111,
    >>>                                   log_dir=log_dir,
    >>>                                   stage='grid+iter',
    >>>                                   model=model,
    >>>                                   verbose=verbose)
    >>> model_fit.fit()
    """

    def __init__(self, 
                 func_files, 
                 TR=1.111, 
                 log_dir=None, 
                 output_base=None, 
                 output_dir=None, 
                 verbose=False, 
                 fit_hrf=False, 
                 standardization="psc", 
                 rsq_threshold=None, 
                 n_pix=100,
                 **kwargs):

        self.func_files         = func_files
        self.verbose            = False
        self.TR                 = TR
        self.log_dir            = log_dir
        self.duration           = 0.25
        self.output_dir         = output_dir
        self.model              = "norm"
        self.stage              = "grid+iter"
        self.verbose            = verbose
        self.fit_hrf            = fit_hrf
        self.standardization    = standardization
        self.rsq_threshold      = rsq_threshold
        self.n_pix              = n_pix

        # try to derive output name from BIDS-components in input file
        if output_base == None:
            try:
                comps = utils.split_bids_components(func_files)
                self.output_base = f"sub-{comps['sub']}_ses-{comps['ses']}_task-{comps['task']}_acq-{comps['acq']}"
            except:
                raise ValueError(f"Could not read BIDS-components from {self.func_files}. Please format accordingly or specify 'output_base'")
        else:
            self.output_base = output_base

        # update kwargs for pRFmodelFitting & plotting
        self.__dict__.update(kwargs)

        # fetch data
        self.prepare_func()

        # fetch design
        self.prepare_design()

        if self.verbose:
            if self.design.shape[-1] == self.data.shape[0]:
                print("Shapes of design matrix and functional data match. Ready for fit!")
            else:
                raise ValueError(f"Shapes of functional data ({self.data.shape[0]}) does not match shape of design matrix ({self.design.shape[-1]}). You might have to transpose your data.")

    def fit(self, **kwargs):

        if self.verbose:
            print("\n---------------------------------------------------------------------------------------------------")
            print(f"STAGE 1 [no HRF]")

        data_fname = opj(self.output_dir, self.output_base+'_desc-data.npy')
        if self.verbose:
            print(f"Saving data as {data_fname}")
        np.save(data_fname, self.data)
        
        if not hasattr(self, "design"):
            self.prepare_design(**kwargs)
        
        # stage 1 - no HRF
        self.stage1 = prf.pRFmodelFitting(self.data.T,
                                          design_matrix=self.design, 
                                          TR=self.TR, 
                                          model=self.model, 
                                          stage=self.stage, 
                                          verbose=self.verbose,
                                          output_dir=self.output_dir,
                                          output_base=self.output_base,
                                          write_files=True,
                                          rsq_threshold=self.rsq_threshold,
                                          **kwargs)

        # stage can be 'grid', 'iter' or 'grid+iter'
        if "iter" in self.stage:
            used_stage = "iter"
        else:
            used_stage = "grid"

        # check if params-file already exists
        self.pars_file = opj(self.output_dir, self.output_base+f"_model-{self.model}_stage-{used_stage}_desc-prf_params.npy")
        # run fit if file doesn't exist, otherwise load them in case we want to fit the prf
        if not os.path.exists(self.pars_file):
            if self.verbose:
                print(f"Running fit with {self.model}-model (HRF=False)")

            # fit
            self.stage1.fit()
        else:
            if self.verbose:
                print(f"Parameter file '{self.pars_file}'")
            
            # load
            self.stage1.load_params(self.pars_file, model=self.model, stage=used_stage)

        # stage2 - fit HRF after initial iterative fit
        if self.fit_hrf:

            if self.verbose:
                print("\n---------------------------------------------------------------------------------------------------")
                print(f"STAGE 2 [HRF]- Running fit with {self.model}-model")

            # check first if we can insert the entire fitter (in case of fitting HRF directly after the other)
            # then check if we have old parameters when running stage1 and stage2 separately
            if hasattr(self.stage1, f"{self.model}_fitter"):
                if self.verbose:
                    print(f"Use '{self.model}_fitter' from {self.stage1} [fitter]")

                prev_fitter = getattr(self.stage1, f"{self.model}_fitter")
                old_pars = None
            elif hasattr(self.stage1, f"{self.model}_{used_stage}"):
                if self.verbose:
                    print(f"Use '{self.model}_{used_stage}' from {self.stage1} [parameters]")

                old_pars = getattr(self.stage1, f"{self.model}_{used_stage}")
                prev_fitter = None
            else:
                raise ValueError(f"Could not derive '{self.model}_fitter' or '{self.model}_{used_stage}' from {self.stage1}")

            # add tag to output to differentiate between HRF=false and HRF=true
            self.output_base += "_hrf-true"

            # initiate fitter object with previous fitter
            self.stage2 = prf.pRFmodelFitting(self.data.T, 
                                              design_matrix=self.design, 
                                              TR=self.stage1.TR, 
                                              model=self.model, 
                                              stage=self.stage, 
                                              verbose=self.verbose,
                                              fit_hrf=True,
                                              output_dir=self.output_dir,
                                              output_base=self.output_base,
                                              write_files=True,                                
                                              previous_gaussian_fitter=prev_fitter,
                                              old_params=old_pars)


        # check if params-file already exists
        self.hrf_file = opj(self.output_dir, self.output_base+f"_model-{self.model}_stage-{used_stage}_desc-prf_params.npy")
        # run fit if file doesn't exist, otherwise load them in case we want to fit the prf
        if not os.path.exists(self.hrf_file):
            if self.verbose:
                print(f"Running fit with {self.model}-model (HRF=False)")

            # fit
            self.stage2.fit()
        else:
            if self.verbose:
                print(f"Parameter file '{self.hrf_file}'")
            
            # load
            self.stage2.load_params(self.hrf_file, model=self.model, stage=used_stage)

    def prepare_design(self):

        if self.verbose:
            print(f"Using {self.log_dir} for design")

        self.design = prf.create_line_prf_matrix(self.log_dir, 
                                                 stim_duration=self.duration, 
                                                 nr_trs=self.data.shape[0],
                                                 stim_at_half_TR=True,
                                                 n_pix=self.n_pix,
                                                 TR=self.TR)

        if self.verbose:
            print(f"Design matrix has shape: {self.design.shape}")

    def prepare_func(self):

        if self.verbose:
            print("Input files:")

            if isinstance(self.func_files, str):
                print(f" {self.func_files}")
            elif isinstance(self.func_files, list):
                for ii in self.func_files:
                    print(f" {ii}")
            else:
                raise ValueError(f"Unknown input type {type(self.func_files)}. Must be a array/string pointing to an existing file or a list of files/arrays")
            
        self.partial = dataset.Dataset(self.func_files,
                                       TR=self.TR,
                                       use_bids=True,
                                       standardization=self.standardization,
                                       verbose=self.verbose)

        # fetch data and filter out NaNs
        self.df_data = self.partial.fetch_fmri()
        self.data = self.df_data.values
        
        if self.verbose:
            print(f"Func data has shape: {self.data.shape} ({type(self.data)})")

class FitLines(dataset.Dataset):
    """FitLines

    Fitting object for line-data that has been reconstructed with https://github.com/gjheij/linescanning/blob/main/bin/call_linerecon (includes NORDIC). This workflow results in *mat-files, which is a format compatible with https://github.com/gjheij/linescanning/blob/main/linescanning/dataset.py#L1170. This class creates the design matrix for the run given a log-directory, applies low/high pass filtering, and fits the pRFs to the data. We'll average runs and iterations.

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
    ribbon: list, optional
        If a list of voxels representing the gray-matter ribbon is specified, we'll perform the pRF-fit only on these voxels. This is generally favourable, but because the ribbon differs across subjects the default is None.
    average: bool, optional
        Average across runs before throwing data into the fitter. Default = True. Allows you to fit data on separate runs as well
    rsq_threshold: float, optional
        Manual set threshold for r2 if the one specified in the `template` is not desired.
    fit_hrf: bool, optional
        Fit the HRF during pRF-fitting. If `True`, the fitting will consist of two stages: first, a regular fitting without HRF estimatiobn. Then, the fitting object of that fit is inserted as `previous_gaussian_fitter` into a new fitter object with HRF estimation turned on. Default = False.

    Raises
    ----------
    ValueError
        If no output basename was specified, and the input file is not formatted according to BIDS
    ValueError
        If the **first** dimension of the functional data does not match the **last** dimension of the design matrix (time)

    Example
    ----------
    >>> func_files = get_file_from_substring(["task-pRF", "bold.mat"], func_dir)
    >>> model_fit = fitting.FitLines(func_files=func_files,
    >>>                              output_dir=output_dir,
    >>>                              TR=0.105,
    >>>                              low_pass=True,
    >>>                              high_pass=True,
    >>>                              window_size=19,
    >>>                              poly_order=3,
    >>>                              log_dir=log_dir,
    >>>                              stage='grid',
    >>>                              model='norm',
    >>>                              verbose=True)
    >>> model_fit.fit()

    Notes
    ----------
    Because we also need to filter our data in this class, we can specify any argument accepted by :class:`linescanning.dataset.Dataset` via `kwargs`.
    """

    def __init__(self, 
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
                 average=True,
                 rsq_threshold=None,
                 fit_hrf=False,
                 n_pix=100,
                 **kwargs):

        self.func_files         = func_files
        self.verbose            = False
        self.TR                 = TR
        self.log_dir            = log_dir
        self.duration           = 0.25
        self.output_dir         = output_dir
        self.model              = "norm"
        self.stage              = "grid+iter"
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

        # try to derive output name from BIDS-components in input file
        if output_base == None:
            try:
                if isinstance(func_files, list):
                    func = func_files[0]
                else:
                    func = func_files

                comps = utils.split_bids_components(func)
                self.output_base = f"sub-{comps['sub']}_ses-{comps['ses']}_task-{comps['task']}_acq-line"
            except:
                raise ValueError(f"Could not read BIDS-components from {func}. Please format accordingly or specify 'output_base'")
        else:
            self.output_base = output_base
        
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
        self.average_iterations(**kwargs)

    def fit(self, **kwargs):

        if self.verbose:
            print("\n---------------------------------------------------------------------------------------------------")
            print(f"STAGE 1 [no HRF]")

        # use data with/without baseline
        if self.strip_baseline:
            if self.verbose:
                print("Using data WITHOUT baseline for fitting")
            self.data_for_fitter = self.avg_iters_no_baseline.copy()
        else:
            if self.verbose:
                print("Using data WITH baseline for fitting")
            self.data_for_fitter = self.avg_iters_baseline.copy()

        data_fname = opj(self.output_dir, self.output_base+'_desc-data.npy')
        if self.verbose:
            print(f"Saving data as {data_fname}")
        np.save(data_fname, self.data_for_fitter)
        
        if not hasattr(self, "design"):
            self.prepare_design(**kwargs)
        
        # stage 1 - no HRF
        self.stage1 = prf.pRFmodelFitting(self.data_for_fitter.T,
                                          design_matrix=self.design, 
                                          TR=self.TR, 
                                          model=self.model, 
                                          stage=self.stage, 
                                          verbose=self.verbose,
                                          output_dir=self.output_dir,
                                          output_base=self.output_base,
                                          write_files=True,
                                          rsq_threshold=self.rsq_threshold,
                                          **kwargs)

        # stage can be 'grid', 'iter' or 'grid+iter'
        if "iter" in self.stage:
            used_stage = "iter"
        else:
            used_stage = "grid"

        # check if params-file already exists
        self.pars_file = opj(self.output_dir, self.output_base+f"_model-{self.model}_stage-{used_stage}_desc-prf_params.npy")
        # run fit if file doesn't exist, otherwise load them in case we want to fit the prf
        if not os.path.exists(self.pars_file):
            if self.verbose:
                print(f"Running fit with {self.model}-model (HRF=False)")

            # fit
            self.stage1.fit()
        else:
            if self.verbose:
                print(f"Parameter file '{self.pars_file}'")
            
            # load
            self.stage1.load_params(self.pars_file, model=self.model, stage=used_stage)

        # stage2 - fit HRF after initial iterative fit
        if self.fit_hrf:

            if self.verbose:
                print("\n---------------------------------------------------------------------------------------------------")
                print(f"STAGE 2 [HRF]- Running fit with {self.model}-model")

            # check first if we can insert the entire fitter (in case of fitting HRF directly after the other)
            # then check if we have old parameters when running stage1 and stage2 separately
            if hasattr(self.stage1, f"{self.model}_fitter"):
                if self.verbose:
                    print(f"Use '{self.model}_fitter' from {self.stage1} [fitter]")

                prev_fitter = getattr(self.stage1, f"{self.model}_fitter")
                old_pars = None
            elif hasattr(self.stage1, f"{self.model}_{used_stage}"):
                if self.verbose:
                    print(f"Use '{self.model}_{used_stage}' from {self.stage1} [parameters]")

                old_pars = getattr(self.stage1, f"{self.model}_{used_stage}")
                prev_fitter = None
            else:
                raise ValueError(f"Could not derive '{self.model}_fitter' or '{self.model}_{used_stage}' from {self.stage1}")

            # add tag to output to differentiate between HRF=false and HRF=true
            self.output_base += "_hrf-true"

            # initiate fitter object with previous fitter
            self.stage2 = prf.pRFmodelFitting(self.data_for_fitter.T, 
                                              design_matrix=self.design, 
                                              TR=self.stage1.TR, 
                                              model=self.model, 
                                              stage=self.stage, 
                                              verbose=self.verbose,
                                              fit_hrf=True,
                                              output_dir=self.output_dir,
                                              output_base=self.output_base,
                                              write_files=True,                                
                                              previous_gaussian_fitter=prev_fitter,
                                              old_params=old_pars)


        # check if params-file already exists
        self.hrf_file = opj(self.output_dir, self.output_base+f"_model-{self.model}_stage-{used_stage}_desc-prf_params.npy")
        # run fit if file doesn't exist, otherwise load them in case we want to fit the prf
        if not os.path.exists(self.hrf_file):
            if self.verbose:
                print(f"Running fit with {self.model}-model (HRF=False)")

            # fit
            self.stage2.fit()
        else:
            if self.verbose:
                print(f"Parameter file '{self.hrf_file}'")
            
            # load
            self.stage2.load_params(self.hrf_file, model=self.model, stage=used_stage)

    def prepare_design(self, **kwargs):
        
        dm_fname = opj(self.output_dir, self.output_base+'_desc-design_matrix.npy')
        if os.path.exists(dm_fname):
            if self.verbose:
                print(f"Using existing design matrix: {dm_fname}")
            
            self.design = io.loadmat(dm_fname)
            tag = list(self.design.keys())[-1]
            self.design = self.design[tag]
        else:
            if self.verbose:
                print(f"Using {self.log_dir} for design")

            self.design = prf.create_line_prf_matrix(self.log_dir, 
                                                     stim_duration=0.25,
                                                     nr_trs=self.avg_iters_baseline.shape[0],
                                                     TR=0.105,
                                                     verbose=self.verbose,
                                                     n_pix=self.n_pix,
                                                     **kwargs)

            # strip design from its baseline
            if self.strip_baseline:
                self.design = self.design[...,self.baseline.shape[0]:]

            # save design matrix for later reference
            if self.verbose:
                print(f"Saving design matrix as {dm_fname}")
            io.savemat(dm_fname, {'stim': self.design})

        # check the shapes of design and functional data match
        if self.verbose:
            print(f"Design matrix has shape: {self.design.shape}")

        if hasattr(self, "data_for_fitter"):
            if self.verbose:
                if self.design.shape[-1] == self.data_for_fitter.shape[0]:
                    print("Shapes of design matrix and functional data match. Ready for fit!")
                else:
                    raise ValueError(f"Shapes of functional data ({self.data_for_fitter.shape[0]}) does not match shape of design matrix ({self.design.shape[-1]}). You might have to transpose your data or cut away baseline.")            
        else:
            print("WARNING: I'm not sure which data you're going to use for fitting; can't verify if the shape of the design matrix matches the functional data.. This is likely because you're running 'prepare_design()' before 'fit()'. You can turn this message off by removing 'prepare_design()', as it's called by 'fit()'")


    def prepare_func(self, **kwargs):

        if self.verbose:
            print("\n---------------------------------------------------------------------------------------------------")
            print(f"Preprocessing functional data")                    
        
        super().__init__(self.func_files, verbose=self.verbose, **kwargs)
        # self.func = dataset.Dataset(self.func_files, verbose=self.verbose, **kwargs)

        # fetch data and filter out NaNs
        self.data = self.fetch_fmri()

        if self.ribbon:
            if self.verbose:
                print(" Including GM-voxels only (saves times)")

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
        
        if self.verbose:
            print(f" Baseline    = {start} vols (~{round(start*self.TR,2)}s) based on TR of {self.TR}s ({self.baseline_duration}s was specified/requested)")
            print(f" 1 iteration = {iter_size} vols (~{round(iter_size*self.TR,2)}s) based on TR of {self.TR}s ({self.iter_duration}s was specified/requested)")

        if self.verbose:
            if self.n_iterations > 1:
                print(f" Chunking/averaging {self.n_iterations} iterations")
            elif self.n_iterations == 1:
                print(f" Averaging of chunks turned OFF (nr of iterations = {self.n_iterations})")
            else:
                raise ValueError(f"Unknown recognized number of iterations: '{self.n_iterations}'")

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

        self.avg_iters_baseline     = np.concatenate((self.baseline, np.concatenate(self.iter_chunks, axis=-1).mean(axis=-1)))
        self.avg_iters_no_baseline  = np.concatenate(self.iter_chunks, axis=-1).mean(axis=-1)

        if self.verbose:
            print(f" With baseline: {self.avg_iters_baseline.shape} | No baseline: {self.avg_iters_no_baseline.shape}")


class pRFResults():

    def __init__(self, prf_params, verbose=False, TR=0.105, **kwargs):

        self.prf_params     = prf_params
        self.verbose        = verbose
        self.TR             = TR
        self.__dict__.update(kwargs)

        if self.verbose:
            print(f"Loading in files:")
            print(f" pRF params:    {self.prf_params}")

        # fetch settings; if list > get the most recent one
        self.yml = utils.get_file_from_substring("settings", os.path.dirname(self.prf_params))
        if isinstance(self.yml, list):
            self.yml = self.yml[-1]

        # the params file should have a particular naming that allows us to read specs:
        self.file_components    = split_params_file(self.prf_params)
        self.model              = self.file_components['model']
        self.stage              = self.file_components['stage']

        try:
            self.run = self.file_components['run']
            search_design = [f"run-{self.file_components['run']}", 'desc-design_matrix']
            search_data = [f"run-{self.file_components['run']}", 'desc-data']
        except:
            self.run = None
            search_design = 'desc-design_matrix'
            search_data = 'desc-data'

        # get design matrix  data
        self.fn_design          = utils.get_file_from_substring(search_design, os.path.dirname(self.prf_params))
        self.fn_data            = utils.get_file_from_substring(search_data, os.path.dirname(self.prf_params))

        # check if design is a numpy-file or mat-file
        if self.fn_design.endswith("npy"):
            self.design = np.load(self.fn_design)
        elif self.fn_design.endswith("mat"):
            self.design = io.loadmat(self.fn_design)
            self.design = self.design[list(self.design.keys())[-1]]
        
        # load data
        self.data = np.load(self.fn_data)

        if self.verbose:
            print(f" Design matrix: {self.fn_design}")
            print(f" fMRI data:     {self.fn_data}")

        # initiate the fitting class
        self.model_fit = prf.pRFmodelFitting(self.data.T,
                                             design_matrix=self.design,
                                             settings=self.yml,
                                             model=self.model,
                                             TR=self.TR)

        # load the parameters
        self.model_fit.load_params(self.prf_params, model=self.model, stage=self.stage, run=self.run)

    def plot_prf_timecourse(self, vox_id=None, vox_range=None, save=True, ext="png", **kwargs):

        if vox_range == None:
            if save:
                fname = opj(os.path.dirname(self.prf_params), f"plot_vox-{vox_id}.{ext}")
            else:
                fname = None
            pars,_,_ = self.model_fit.plot_vox(vox_nr=vox_id, 
                                               model=self.model,
                                               transpose=False, 
                                               axis_type="time",
                                               save_as=fname,
                                               **kwargs)

        else:
            for vox_id in range(*vox_range):
                if save:
                    fname = opj(os.path.dirname(self.prf_params), f"plot_vox-{vox_id}.{ext}")
                else:
                    fname = None

                pars,_,_ = self.model_fit.plot_vox(vox_nr=vox_id, 
                                                   model=self.model,
                                                   transpose=False, 
                                                   axis_type="time",
                                                   save_as=fname,
                                                   **kwargs)
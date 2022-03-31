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

    def __init__(self, func_files, TR=1.111, log_dir=None, output_base=None, output_dir=None, verbose=False, **kwargs):

        self.func_files         = func_files
        self.verbose            = False
        self.TR                 = TR
        self.log_dir            = log_dir
        self.duration           = 0.25
        self.output_dir         = output_dir
        self.model              = "norm"
        self.stage              = "grid+iter"
        self.verbose            = verbose

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

    def fit(self):
        
        if self.verbose:
            print(f"Running fit with {self.model}-model")

        fitter = prf.pRFmodelFitting(self.data.T, 
                                     design_matrix=self.design, 
                                     TR=self.TR, 
                                     model=self.model, 
                                     stage=self.stage, 
                                     verbose=self.verbose,
                                     output_dir=self.output_dir,
                                     output_base=self.output_base,
                                     write_files=True)

        fitter.fit()

    def prepare_design(self):

        if self.verbose:
            print(f"Using {self.log_dir} for design")

        self.design = prf.create_line_prf_matrix(self.log_dir, 
                                                 stim_duration=self.duration, 
                                                 nr_trs=self.data.shape[0],
                                                 stim_at_half_TR=True,
                                                 TR=self.TR)

        if self.verbose:
            print(f"Design matrix has shape: {self.design.shape}")

    def prepare_func(self):

        if self.verbose:
            print(f"Received {self.func_files}")

        self.partial = dataset.Dataset(self.func_files,
                                      TR=self.TR,
                                      deleted_first_timepoints=0, 
                                      deleted_last_timepoints=0,
                                      lb=0.01,
                                      high_pass=True,
                                      low_pass=False,
                                      window_size=5,
                                      use_bids=True,
                                      verbose=self.verbose)

        # fetch data and filter out NaNs
        self.data = self.partial.fetch_fmri()
        self.data = utils.filter_for_nans(self.data.values)
        
        if self.verbose:
            print(f"Func data has shape: {self.data.shape}")

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
                 fmri_output="zscore",
                 average=True,
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
        self.fmri_output        = fmri_output
        self.average            = average

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

        # update kwargs for pRFmodelFitting & plotting
        self.__dict__.update(kwargs)

        if self.log_dir == None:
            raise ValueError(f"Please specify a log-directory..")
        
        # fetch data
        self.prepare_func(**kwargs)

        # average iterations or assume experiment is 1 iteration
        self.average_iterations(**kwargs)

    def fit(self, **kwargs):
    
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
            self.prepare_design()
        
        if self.verbose:
            print(f"Running fit with {self.model}-model")

        self.fitter = prf.pRFmodelFitting(self.data_for_fitter.T,
                                          design_matrix=self.design, 
                                          TR=self.TR, 
                                          model=self.model, 
                                          stage=self.stage, 
                                          verbose=self.verbose,
                                          output_dir=self.output_dir,
                                          output_base=self.output_base,
                                          write_files=True,
                                          **kwargs)

        self.fitter.fit()

    def prepare_design(self):

        if self.verbose:
            print(f"Using {self.log_dir} for design")

        self.design = prf.create_line_prf_matrix(self.log_dir, 
                                                 stim_duration=0.25,
                                                 nr_trs=self.avg_iters_baseline.shape[0],
                                                 TR=0.105,
                                                 verbose=self.verbose)

        if self.strip_baseline:
            self.design = self.design[...,self.baseline.shape[0]:]

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

        dm_fname = opj(self.output_dir, self.output_base+'_desc-design_matrix.npy')
        if self.verbose:
            print(f"Saving design matrix as {dm_fname}")
        io.savemat(dm_fname, {'stim': self.design})

    def prepare_func(self, **kwargs):

        super().__init__(self.func_files, verbose=self.verbose, **kwargs)
        # self.func = dataset.Dataset(self.func_files, verbose=self.verbose, **kwargs)

        # fetch data and filter out NaNs
        self.data = self.fetch_fmri(dtype=self.fmri_output)
        self.avg = self.data.groupby(['subject', 't']).mean()

        if self.ribbon != None:
            self.ribbon_idc = list(np.arange(*self.ribbon))

            if self.verbose:
                print(f"Selecting GM-voxels: {self.ribbon}")

            self.df_ribbon = utils.select_from_df(self.avg, expression='ribbon', indices=self.ribbon_idc)

    def average_iterations(self, **kwargs):

        # initialize Dataset-parent
        if not hasattr(self, 'avg'):
            self.prepare_func(**kwargs)

        # fitting on ribbon voxels is infinitely faster
        if hasattr(self, "df_ribbon"):
            use_data = self.df_ribbon.copy()
        else:
            use_data = self.avg.copy()

        # check if we should remove initial volumes (no advisable)
        if not hasattr(self, "deleted_first_timepoints"):
            start = int(round(self.baseline_duration/self.TR, 0))
        else:
            start = int(round(self.baseline_duration/self.TR, 0)) - int(round(self.deleted_first_timepoints*self.TR, 0))

        # fetch baseline volumes
        self.baseline = use_data[:start]
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
            if start+iter_size < use_data.shape[0]:
                chunk = use_data.values[start:start+iter_size]
            else:
                chunk = use_data.values[start+iter_size:]
                padded_array = np.zeros((iter_size, use_data.shape[-1]))
                padded_array[:chunk.shape[0]] = chunk
                chunk = padded_array.copy()

            self.iter_chunks.append(chunk[...,np.newaxis])
            start += iter_size

        self.avg_iters_baseline     = np.concatenate((self.baseline, np.concatenate(self.iter_chunks, axis=-1).mean(axis=-1)))
        self.avg_iters_no_baseline  = np.concatenate(self.iter_chunks, axis=-1).mean(axis=-1)

        if self.verbose:
            print(f" With baseline: {self.avg_iters_baseline.shape} | No baseline: {self.avg_iters_no_baseline.shape}")


class pRFResults():

    def __init__(self, prf_params, verbose=False, **kwargs):

        self.prf_params     = prf_params
        self.verbose        = verbose
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

        # get design matrix  data
        self.fn_design          = utils.get_file_from_substring([f"run-{self.file_components['run']}", 'desc-design_matrix'], os.path.dirname(self.prf_params))
        self.fn_data            = utils.get_file_from_substring([f"run-{self.file_components['run']}", 'desc-data'], os.path.dirname(self.prf_params))
        self.design             = io.loadmat(self.fn_design)['stim']
        self.data               = np.load(self.fn_data)

        if self.verbose:
            print(f" Design matrix: {self.fn_design}")
            print(f" fMRI data:     {self.fn_data}")

        # initiate the fitting class
        self.model_fit = prf.pRFmodelFitting(self.data.T,
                                             design_matrix=self.design,
                                             settings=self.yml,
                                             model=self.model)

        # load the parameters
        self.model_fit.load_params(np.load(self.prf_params), model=self.model, stage=self.stage)

            

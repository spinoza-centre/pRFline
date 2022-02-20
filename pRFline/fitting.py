from linescanning import utils, prf, dataset
import os
from nipype.interfaces import fsl, ants, freesurfer
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


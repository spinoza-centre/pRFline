from linescanning import utils, prf, dataset
import os
from nipype.interfaces import fsl, ants, freesurfer
opj = os.path.join

class FitPartialFOV:

    def __init__(self, subject, func_files=None, trafo_file=None, TR=1.111, log_dir=None, output_base=None, output_dir=None, verbose=False, **kwargs):


        self.subject            = subject
        self.trafo_file         = trafo_file
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
            print(f"Running fit")

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


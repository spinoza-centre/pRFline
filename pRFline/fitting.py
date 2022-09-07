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
from pRFline.utils import split_params_file
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
    ribbon: list, optional
        If a list of voxels representing the gray-matter ribbon is specified, we'll perform the pRF-fit only on these voxels. This is generally favourable, but because the ribbon differs across subjects the default is None.
    average: bool, optional
        Average across runs before throwing data into the fitter. Default = True. Allows you to fit data on separate runs as well
    rsq_threshold: float, optional
        Manual set threshold for r2 if the one specified in the `template` is not desired.
    fit_hrf: bool, optional
        Fit the HRF during pRF-fitting. If `True`, the fitting will consist of two stages: first, a regular fitting without HRF estimatiobn. Then, the fitting object of that fit is inserted as `previous_gaussian_fitter` into a new fitter object with HRF estimation turned on. Default = False.
    is_lines: bool, optional
        There's a slightly different procedure for line-scanning compared to whole-brain or 3D-EPI: for the line-scanning, we have two iterations per run. So we need to average those in order to get a single iteration. By default True

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
        average=True,
        rsq_threshold=None,
        fit_hrf=False,
        n_pix=100,
        design_only=False,
        is_lines=True,
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
        self.stage1 = prf.pRFmodelFitting(
            self.data_for_fitter.T,
            design_matrix=self.design, 
            TR=self.TR, 
            model=self.model, 
            stage=self.stage, 
            verbose=self.verbose,
            output_dir=self.output_dir,
            output_base=self.output_base,
            write_files=True,
            rsq_threshold=self.rsq_threshold,
            fix_bold_baseline=True,
            **kwargs)

        # check if params-file already exists
        self.pars_file = opj(self.output_dir, self.output_base+f"_model-{self.model}_stage-{self.stage}_desc-prf_params.pkl")
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
            self.stage1.load_params(self.pars_file, model=self.model, stage=self.stage)

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
            elif hasattr(self.stage1, f"{self.model}_{self.stage}"):
                if self.verbose:
                    print(f"Use '{self.model}_{self.stage}' from {self.stage1} [parameters]")

                old_pars = getattr(self.stage1, f"{self.model}_{self.stage}")
                prev_fitter = None
            else:
                raise ValueError(f"Could not derive '{self.model}_fitter' or '{self.model}_{self.stage}' from {self.stage1}")

            # add tag to output to differentiate between HRF=false and HRF=true
            self.output_base += "_hrf-true"

            # initiate fitter object with previous fitter
            self.stage2 = prf.pRFmodelFitting(
                self.data_for_fitter.T, 
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
                fix_bold_baseline=True,
                rsq_threshold=self.rsq_threshold,
                old_params=old_pars)

        # check if params-file already exists
        
        self.hrf_file = opj(self.output_dir, self.output_base+f"_model-{self.model}_stage-{self.stage}_desc-prf_params.pkl")
        # run fit if file doesn't exist, otherwise load them in case we want to fit the prf
        if not os.path.exists(self.hrf_file):
            if self.verbose:
                print(f"Running fit with {self.model}-model (HRF=True)")

            # fit
            self.stage2.fit()
        else:
            if self.verbose:
                print(f"Parameter file '{self.hrf_file}'")
            
            # load
            self.stage2.load_params(self.hrf_file, model=self.model, stage=self.stage)

    def prepare_design(self, **kwargs):
        
        dm_fname = opj(self.output_dir, self.output_base+'_desc-design_matrix.mat')
        if os.path.exists(dm_fname):
            if self.verbose:
                print(f"Using existing design matrix: {dm_fname}")
            
            self.design = prf.read_par_file(dm_fname)
        else:
            if self.verbose:
                print(f"Using {self.log_dir} for design")

            self.design = prf.create_line_prf_matrix(
                self.log_dir, 
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
            io.savemat(dm_fname, {"stim": self.design})

        # check the shapes of design and functional data match
        if self.verbose:
            print(f"Design matrix has shape: {self.design.shape}")

        if hasattr(self, "data_for_fitter"):
            if self.verbose:
                if self.design.shape[-1] == self.data_for_fitter.shape[0]:
                    print("Shapes of design matrix and functional data match. Ready for fit!")
                else:
                    raise ValueError(f"Shapes of functional data ({self.data_for_fitter.shape[0]}) does not match shape of design matrix ({self.design.shape[-1]}). You might have to transpose your data or cut away baseline.")

    def prepare_func(self, **kwargs):

        if self.verbose:
            print("\n---------------------------------------------------------------------------------------------------")
            print(f"Preprocessing functional data")                    
        
        super().__init__(self.func_files, verbose=self.verbose, **kwargs)

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

        # the params file should have a particular naming that allows us to read specs:
        self.file_components = split_params_file(self.prf_params)
        self.model = self.file_components['model']
        self.stage = self.file_components['stage']

        # start off with defaults
        self.run = None
        self.acq = None
        search_design = ['desc-design_matrix']
        search_data = ['desc-data']

        # update by checking file components
        for el in ['run', 'acq']:
            try:
                setattr(self, el, self.file_components[el])
                search_design = search_design + [f"{el}-{self.file_components[el]}"]
                search_data = search_data + [f"{el}-{self.file_components[el]}"]
            except:
                pass 

        # get design matrix  data
        self.fn_design = utils.get_file_from_substring(search_design, os.path.dirname(self.prf_params))
        self.fn_data = utils.get_file_from_substring(search_data, os.path.dirname(self.prf_params))

        if self.verbose:
            print(f" Design matrix: {self.fn_design}")
            print(f" fMRI data:     {self.fn_data}")

        # check if design is a numpy-file or mat-file
        self.design = prf.read_par_file(self.fn_design)
        
        # load data
        self.data = np.load(self.fn_data)

        # initiate the fitting class
        self.model_fit = prf.pRFmodelFitting(
            self.data.T,
            design_matrix=self.design,
            model=self.model,
            TR=self.TR)

        # load the parameters
        self.model_fit.load_params(self.prf_params, model=self.model, stage=self.stage)

    def plot_prf_timecourse(
        self, 
        vox_nr=None, 
        vox_range=None, 
        save=False, 
        save_dir=None, 
        ext="png", 
        overlap=True, 
        **kwargs):

        if vox_range == None:
            vox_range = [vox_nr,vox_nr+1]

        self.voxel_data = {}
        for vox_id in range(*vox_range):

            if save:
                if save_dir == None:
                    save_dir = os.path.dirname(self.prf_params)
                
                fname = opj(save_dir, f"plot_vox-{vox_id}.{ext}")
            else:
                fname = None

            self.voxel_data[vox_id] = self.model_fit.plot_vox(
                vox_nr=vox_id, 
                model=self.model,
                transpose=False, 
                axis_type="time",
                save_as=fname,
                resize_pix=270,
                title='pars',
                **kwargs)

        # target pRF
        self.target_obj = prf.CollectSubject(
            subject=f"sub-{self.file_components['sub']}",
            derivatives=os.environ.get('DIR_DATA_DERIV'),
            model=self.model,
            best_vertex=True)

        if save:
            if save_dir == None:
                save_dir = os.path.dirname(self.prf_params)

            fname = opj(save_dir, f"plot_vox-target.{ext}")
        else:
            fname = None

        self.target_obj.target_prediction_prf(save_as=fname, **kwargs)

        # plot overlap with target vertex
        if overlap:

            fig = plt.figure(figsize=(len(self.voxel_data)*6,6))
            gs = fig.add_gridspec(1,len(self.voxel_data))

            for ix, vox in enumerate(self.voxel_data):
                pars = self.voxel_data[vox][0]
                prf_line = self.voxel_data[vox][1]

                # get target pRF from ses-1
                prf_target = self.target_obj.targ_prf.copy()

                # create different colormaps
                colors = ["#DE3163", "#6495ED"]
                cmap1 = utils.make_binary_cm(colors[0])
                cmap2 = utils.make_binary_cm(colors[1])
                cmaps = [cmap1, cmap2]
                
                # get distance of pRF-centers
                dist = prf.distance_centers(self.target_obj.targ_pars, pars)

                # initiate and plot figure
                axs = fig.add_subplot(gs[ix])
                for ix, obj in enumerate([prf_target,prf_line]):
                    plotting.LazyPRF(
                        obj, 
                        vf_extent=self.target_obj.settings['vf_extent'], 
                        ax=axs, 
                        cmap=cmaps[ix], 
                        cross_color='k', 
                        alpha=0.5, 
                        title=f"Distance = {round(dist, 2)} dva",
                        font_size=30,
                        shrink_factor=0.9,
                        **kwargs)

            # create custom legend
            legend_elements = [
                Line2D([0],[0], marker='o', color='w', label='target pRF', mfc=colors[0], markersize=24, alpha=0.3),
                Line2D([0],[0], marker='o', color='w', label='line pRF', mfc=colors[1], markersize=24, alpha=0.3)]

            L = fig.legend(handles=legend_elements, frameon=False, fontsize=26, loc='lower right')
            plt.setp(L.texts, family='Arial')
            plt.tight_layout()

            if save:
                # save img
                if save_dir == None:
                    save_dir = os.path.dirname(self.prf_params)
                
                img = opj(save_dir, f'prf_overlap.{ext}')
                print(f"Writing {img}")
                fig.savefig(img, bbox_inches='tight')
            else:
                plt.show()

    def plot_depth(
        self, 
        vox_range=[359,365], 
        measures='all', 
        cmap='viridis', 
        ci_color="#cccccc", 
        save=False, 
        save_dir=None, 
        ext="png", 
        **kwargs):
        
        pars = prf.SizeResponse.parse_normalization_parameters(prf.read_par_file(self.prf_params))
        if isinstance(measures, str):
            if measures == "all":
                measures = list(pars.columns)
            else:
                measures = [measures]

        # filter for range
        incl_voxels = list(np.arange(vox_range[0],vox_range[1]))
        range_pars = pars.iloc[incl_voxels,:]
        colors = sns.color_palette(cmap, range_pars.shape[0])

        fig = plt.figure(figsize=(len(measures)*8,8))
        gs = fig.add_gridspec(1,len(measures))

        for ix, par in enumerate(measures):

            if par == "prf_size":
                x_label = "pRF size (dva)"
            elif par == "A":
                x_label = "activation amplitude (A)"
            elif par == "B":
                x_label = "activation constant (B)"
            elif par == "C":
                x_label = "normalization amplitude (C)"
            elif par == "D":
                x_label = "normalization constant (D)"
            elif par == "r2":
                x_label = "variance (r2)"
            else:
                x_label = par
            

            axs = fig.add_subplot(gs[ix])
            vals = range_pars[par].values
            cf = CurveFitter(vals, order=2, verbose=False)

            for idx, ii in enumerate(vals):
                axs.plot(cf.x[idx], ii, 'o', color=colors[idx])

            plotting.LazyPlot(
                cf.y_pred_upsampled,
                axs=axs,
                xx=cf.x_pred_upsampled,
                error=cf.ci_upsampled,
                color=ci_color,
                x_label="depth",
                y_label=x_label,
                **kwargs
            )

        plt.tight_layout()

        if save:
            # save img
            if save_dir == None:
                save_dir = os.path.dirname(self.prf_params)
            
            img = opj(save_dir, f'prf_depth.{ext}')
            print(f"Writing {img}")
            fig.savefig(img, bbox_inches='tight')
        else:
            plt.show()            
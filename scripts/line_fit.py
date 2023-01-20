#!/usr/bin/env python
#$ -q long.q
#$ -cwd
#$ -j Y
#$ -o ../logs
#$ -V

import ast
import os
import sys, getopt
from bids import BIDSLayout
from pRFline import fitting,utils
from linescanning.utils import get_file_from_substring, string2list
import yaml
opj = os.path.join

def main(argv):

    """line_fit.py

    Preprocess and fit the pRF-data using the line-scanning acquisition. Assumes you have aligned your individual slices to one another and saved the transformation files as txt-files in the 'anat'-folder with your slices. 
    If not, it'll throw an error. Basic procedures performed are:
        - Highpass filtering [DCT]
        - aCompCor (if not enough voxels are present for PCA, the individual timecourses of the CSF/WM voxels themselves are used)
        - Lowpass filtering (if specified) [Savitsky-Golay]
        - Voxel selection across runs using average of tissue probabilities
        - Averaging of timecourses across runs and iterations
        - Perform fitting

    Parameters
    ----------
    -b|--bids_dir <bids_dir>        path where 'func' and 'anat' folders live
    -o|--output_dir <output_dir>    output directory + basename for output; some stuff about model type, stage, 
                                    and parameter type will be appended
    -l|--log_dir <log_dir>          directory that contains the "Screenshot"-directory to create the design matrix
    -r|--run <run ID>               run identifier in case we don't want to average across runs and fit the pRFs 
                                    for individual runs
    -i|--n_iter <iterations>        the experiment consists of one set of 8 sweeps (1 iteration) that can be 
                                    repeated multiple times. We can average these iterations by specifying how many iterations we used. By default, the number of iterations will be read from the log-file from ExpTools, which will always be 1 given that we use the directory that has the screenshots. For that, we only need 1 iterations. With this extra flag you can specify how many iterations to consider.
    --rsq <float>                   r2-threshold to use during pRF-fitting. Parameters of voxels < threshold will 
                                    be set to 0
    --ses_trafo <file>              transformation mapping ses-1 to the closest partial anatomy prior to the 
                                    corresponding slice of the line-scanning acquisition
    -h|--help                       print this information
    -q|--qa|--no-fit                do quality control, not full fitting; plots will be stored in the 'anat' 
                                    folder. Stop *before* creation of design matrix (see `--dm` for quitting process *after* design matrix)
    --filter_pca <float>            cutoff frequency for highpass filtering of PCA components. This procedure can 
                                    ensure that task-related frequencies in the PCA-components do not get removed
    --dm                            stop process after making the design matrix. One step further compared to `-q|
                                    --qa` or `--no-fit` which stop *before* the creation of design matrix
    --hrf                           fit the HRF with the pRF-parameters as implemented in `prfpy`    
    -g                              run model fitter with gray matter voxels only (based on the average tissue 
                                    probabilities across runs)
    --ribbon                        Include only voxels in this range. Voxels are read from `class:pRFline.utils. 
                                    SubjectsDict()`
    -v|--verbose                    turn on verbose
    --feed_avg                      Used in combination with `--ribbon`. Tells the fitter to fit stuff first on 
                                    the average of the ribbon, then feed that into individual fitting of ribbon voxels
    --fix <indices>                 Fix certain parameters. Generally used in combination with `--ribbon` and 
                                    `--feed_avg`, where you can fix parameters from the average fit. These must be comma-separated values reflecting indices of the parameter space. E.g., 'x'=0, 'y'=1, 'size'=2, etc. Note that you must know which parameters corresponds to which index for this to succeed. 
    --no_grid                       Don't save grid-parameters; can save clogging up of directories. Default = 
                                    False
    --ow|--overwrite                Overwrite existing files; otherwise files found along the way will be inserted 
                                    in their respective spots
    --lbgfs|--bgfs                  Use L-BGFS minimization, rather than trust-constraint
    --norm                          Fit normalization parameters [default]
    --gauss                         Fit Gaussian parameters
    --dog                           Fit Difference-of-Gaussian parameters
    --css                           Fit CSS parameters

    Returns
    ----------
    Output as per https://linescanning.readthedocs.io/en/latest/classes/prf.html#linescanning.prf.pRFmodelFitting

    Example
    ----------
    >>> # simplest form > will assume certain folders/files to exist
    >>> python line_fit.py -s sub-001 -i 2 --lp --verbose

    >>> # set some variables
    >>> sub=007
    >>> ses=2
    >>> iters=2
    >>> bids_dir=${DIR_DATA_HOME}/sub-${sub}/ses-${ses}
    >>> out_dir=${DIR_DATA_DERIV}/prf/sub-${sub}/ses-${ses}/sub-${sub}_ses-${ses}_task-pRF
    >>> log_dir=log_dir=${DIR_DATA_SOURCE}/sub-${sub}/ses-${ses}/sub-${sub}_ses-${ses}_task-pRF_run-imgs
    >>> ses_trafo=${DIR_DATA_DERIV}/pycortex/sub-${sub}/transforms/sub-${sub}_from-ses1_to-ses${ses}_rec-motion1_desc-genaff.mat

    >>> # run locally, filter PCA-components and verbose
    >>> python line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} -v 

    >>> # run locally, filter PCA-components, verbose, fit only GM-voxels
    >>> python line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} -v -g

    >>> # run QA-only with verbose
    >>> python line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} -v -q

    >>> # submit to cluster
    >>> qsub -N prf_${sub} -pe smp 1 -wd /data1/projects/MicroFunc/Jurjen/programs/project_repos/pRFline/logs line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} -v
    """

    subject         = None
    bids_dir        = None
    run_id          = None
    output_dir      = None
    log_dir         = None
    verbose         = False
    model           = "norm"
    ses_trafo       = None
    run_trafos      = None
    n_iter          = None
    filter_pca      = 0.2
    rsq_thresh      = 0
    qa              = False
    gm_only         = False
    fit_hrf         = False
    filter_strat    = "hp"
    make_report     = True
    do_acompcor     = True
    n_pix           = 100
    stop_at_dm      = False
    t_r             = 0.105
    ribbon          = False
    fix_parameters  = None
    avg_to_ribbon   = False
    save_grid       = False
    overwrite       = False
    full_design     = False
    trim_end        = 300
    strip_baseline  = True
    shift           = 0
    window_length   = None
    poly_order      = None
    constr          = "tc"

    # long options without argument: https://stackoverflow.com/a/54170513
    try:
        opts = getopt.getopt(argv,"nqgvh:b:d:r:o:f:l:i:s:",["help", "bids_dir=", "n_iter=", "lowpass", "ses_trafo=", "output_dir=", "log_dir=", "filter_pca=", "rsq=", "run=", "hrf", "no_report", "verbose", "no_acompcor", "qa", "n_pix=", "dm", "no_fit", "tr=", "ribbon", "fix=", "feed_avg", "save_grid", "overwrite", "ow", "gauss", "norm", "css", "dog", "full_design", "trim_end=", "keep_baseline", "shift=","lp","lbgfs","bgfs"])[0]
    except getopt.GetoptError:
        print("ERROR while handling arguments.. Did you specify an 'illegal' argument..?")
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts: 
        if opt in ("-h", "--help"):
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--subject"):
            subject = arg            
        elif opt in ("-b", "--bids_dir"):
            bids_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-l", "--log_dir"):
            log_dir = arg
        elif opt in ("-r", "--run"):
            run_id = arg
        elif opt in ("-m", "--model"):
            model = arg            
        elif opt in ("-i", "--n_iter"):
            n_iter = arg        
        elif opt in ("--ses_trafo"):
            ses_trafo = arg
        elif opt in ("-n", "--lowpass"):
            filter_strat = "lp"
        elif opt in ("--filter_pca"):
            filter_pca = float(arg)
        elif opt in ("--rsq"):
            rsq_thresh = arg
        elif opt in ("--hrf"):
            fit_hrf = True                 
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-g"):
            gm_only = True
        elif opt in ("-q","--qa","--no_fit"):
            qa = True
        elif opt in ("--no_report"):
            make_report = False
        elif opt in ("--no_acompcor"):
            do_acompcor = False
        elif opt in ("--n_pix"):
            n_pix = arg
        elif opt in ("--dm"):
            stop_at_dm = True
        elif opt in ("--tr"):
            t_r = float(arg)
        elif opt in ("--ribbon"):
            ribbon = True
        elif opt in ("--lp"):
            filter_strat = "lp"
            poly_order = 3
        elif opt in ("--shift"):
            shift = arg
            if "," in shift:
                shift = [float(i) for i in string2list(shift)]
            else:
                shift = float(shift)
        elif opt in ("--fix"):
            fix_parameters = ast.literal_eval(arg)
            if isinstance(fix_parameters, int):
                fix_parameters = [fix_parameters]
            else:
                fix_parameters = list(fix_parameters)          
        elif opt in ("--feed_avg"):
            avg_to_ribbon = True
        elif opt in ("--save_grid"):
            save_grid = True
        elif opt in ("--ow", "--overwrite"):
            overwrite = True
        elif opt in ("--gauss"):
            model = "gauss"
        elif opt in ("--norm"):
            model = "norm"
        elif opt in ("--css"):
            model = "css"
        elif opt in ("--dog"):
            model = "dog"
        elif opt in ("--full_design"):
            full_design = True
        elif opt in ("--trim_end"):
            trim_end = int(arg)   
        elif opt in ("--lbgfs","--bgfs"):
            constr = "bgfs"

    if len(argv) == 0:
        print(main.__doc__)
        sys.exit()

    #---------------------------------------------------------------------------------------
    # get some defaults 
    SD = utils.SubjectsDict()
    exclude = SD.get_exclude(subject)
    ribbon_v = SD.get_ribbon(subject)
    ses = SD.get_session(subject)
    use_bounds = SD.get_bounds(subject)

    if not isinstance(bids_dir, str):
      bids_dir = opj(
        os.environ.get("DIR_DATA_HOME"),
        subject,
        f"ses-{ses}")

    if not isinstance(ses_trafo, str):
      trafo_search = f"from-ses1_to-ses{ses}_rec-motion1"
      trafo_dir = opj(os.environ.get("DIR_DATA_DERIV"), 'pycortex', subject, 'transforms')
      ses_trafo = get_file_from_substring(trafo_search, trafo_dir, exclude=f"._{subject}")

      if isinstance(ses_trafo, list):
        raise ValueError(f"Found mulitple instances for 'ses{ses}_rec-motion1' in '{trafo_dir}': {ses_trafo}")

    if not isinstance(output_dir, str):
      output_dir = opj(
        os.environ.get("DIR_DATA_DERIV"), 
        "prf", 
        subject, 
        f"ses-{ses}", 
        f"{subject}_ses-{ses}_task-pRF")

    if not isinstance(log_dir, str):
      log_dir = opj(
        os.environ.get("DIR_DATA_SOURCE"), 
        subject, 
        f"ses-{ses}", 
        f"{subject}_ses-{ses}_task-pRF_run-imgs")

    # get layout
    layout = BIDSLayout(bids_dir, validate=False)
    anats = layout.get(datatype="anat", return_type="file")
    funcs = layout.get(datatype="func", return_type="file")

    # get run-to-run transformation files
    run_trafos = layout.get(datatype="anat", extension=['txt'], return_type="file")

    #---------------------------------------------------------------------------------------
    # check if we should do single runs or all runs at once for the output name of aCompCor figure
    if run_id != None and run_id != "all":
        func_search = ["task-pRF", "bold.mat", f"run-{run_id}"]
        anat_search = ["acq-1slice", ".nii.gz", f"run-{run_id}"]
    else:
        func_search = ["task-pRF", "bold.mat"]
        anat_search = ["acq-1slice", ".nii.gz"]

    func_files = get_file_from_substring(func_search, funcs, exclude="acq-3DEPI")
    ref_slices = get_file_from_substring(anat_search, anats, exclude="._")

    # transform string input to list as well
    if isinstance(func_files, str):
        func_files = [func_files]

    if isinstance(ref_slices, str):
        ref_slices = [ref_slices]

    if isinstance(run_trafos, str):
        run_trafos = [run_trafos]                

    print("\n---------------------------------------------------------------------------------------------------")
    print(f"line_prf.py - pRF fitting on line-scanning data")

    if not ribbon:
      ribbon_v = None
    
    # excluded runs
    if isinstance(exclude, list):
        print(f"Excluding run(s): {exclude}")
        make_report = False
        for list_type,ll in zip(["funcs","anats","trafos"], [func_files,ref_slices,run_trafos]):
            if isinstance(ll, list):
                for run in exclude:
                    for ff in ll:
                        if list_type == "trafos":
                            target = f"run{run}.txt"
                        else:
                            target = f"run-{run}" 

                        if target in ff:
                            ll.remove(ff)

    # print list of inputs
    print(*func_files, sep="\n")
    
    # check length if lists
    if len(func_files) != len(ref_slices):
        print(*ref_slices, sep="\n")
        raise ValueError(f"number of func files ({len(func_files)}) does not match number of anatomical files ({len(ref_slices)})")

    # check length if lists
    if do_acompcor:
        if len(ref_slices) != 0:
            if len(run_trafos) != len(ref_slices):      
                raise ValueError(f"number of transformation files ({len(run_trafos)}) does not match number of anatomical files ({len(ref_slices)})")

    #---------------------------------------------------------------------------------------
    # read some settings from the log-file
    log_file = get_file_from_substring(".yml", log_dir)
    with open(log_file, 'r', encoding='utf8') as f_in:
        settings = yaml.safe_load(f_in)

    # reconstruct design in LineExps/lineprf/session.py
    design = settings['design']
    stimuli = settings['stimuli']
    
    # old version
    if not "bar_widths" in list(stimuli.keys()):
        sweep_trials    = int(design.get('bar_steps')*2 + (design.get('inter_sweep_blank')//design.get('stim_duration')))
        rest_trials     = int(design.get('inter_sweep_blank')//design.get('stim_duration'))
        block_trials    = int(sweep_trials*2 + rest_trials)
        part_trials     = int(block_trials*2 + rest_trials*2)
        iter_trials     = int(part_trials*design.get('stim_repetitions'))
        t_r             = design.get('repetition_time')
        iter_duration   = iter_trials*design.get('stim_duration')
    else:
        # new version
        bar_dirs            = stimuli.get("bar_directions")
        bar_widths          = stimuli.get("bar_widths")
        bar_steps           = design.get("bar_steps")
        stim_duration       = design.get("stim_duration")
        inter_sweep_blank   = design.get("inter_sweep_blank")
        nr_sweeps           = len(bar_dirs*len(bar_widths))

        iter_duration = nr_sweeps*(stim_duration*bar_steps)+(nr_sweeps*inter_sweep_blank)

    # check if we got n_iter as argument:
    if n_iter == None:
        n_iter = design.get('stim_repetitions')
    else:
        n_iter = int(n_iter)

    # check if we should skip the baseline; should be skipped if n_iter > 1 because of weird artifacts of averaging
    if n_iter > 1:
        if not full_design or strip_baseline:
            strip_baseline = True

    #---------------------------------------------------------------------------------------
    # set output basename
    if run_id != None:
        output_base_prf = os.path.basename(output_dir)+f"_run-{run_id}"
    else:
        if len(func_files) > 1:
            output_base_prf = os.path.basename(output_dir)+f"_run-avg"
        else:
            output_base_prf = os.path.basename(output_dir)

    #---------------------------------------------------------------------------------------
    # initiate model fitting
    model_fit = fitting.FitpRFs(
        func_files=func_files,
        TR=t_r,
        filter_strategy=filter_strat,
        log_dir=log_dir,
        stage='iter',
        model=model,
        baseline_duration=design.get('start_duration'),
        iter_duration=iter_duration,
        n_iterations=n_iter,
        verbose=verbose,
        strip_baseline=strip_baseline,
        acompcor=do_acompcor,
        ref_slice=ref_slices,
        filter_pca=filter_pca,
        rsq_threshold=rsq_thresh,
        ses1_2_ls=ses_trafo,
        run_2_run=run_trafos,
        output_dir=os.path.dirname(output_dir),
        output_base=output_base_prf,
        voxel_cutoff=300,
        ribbon=ribbon_v,
        gm_only=gm_only,
        fix_parameters=fix_parameters,
        fit_hrf=fit_hrf,
        report=make_report,
        n_pix=n_pix,
        design_only=stop_at_dm,
        start_avg=avg_to_ribbon,
        is_lines=True,
        save_grid=save_grid,
        overwrite=overwrite,
        full_design=full_design,
        deleted_last_timepoints=trim_end,
        shift=shift,
        window_length=window_length,
        poly_order=poly_order,
        constraints=constr,
        use_grid_bounds=use_bounds)

    # fit
    if not qa:
        model_fit.fit()

if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python
#$ -q long.q@jupiter
#$ -cwd
#$ -j Y
#$ -V

import os
import sys, getopt
from pRFline import fitting
from linescanning.utils import get_file_from_substring
import yaml
opj = os.path.join

def main(argv):

    """line_fit.py

    Preprocess and fit the pRF-data using the line-scanning acquisition. Assumes you have aligned your individual slices to one another and saved the transformation files as txt-files in the 'anat'-folder with your slices. If not, it'll throw an error. Basic procedures performed are:
        - Highpass filtering [DCT]
        - aCompCor (if not enough voxels are present for PCA, the individual timecourses of the CSF/WM voxels themselves are used)
        - Lowpass filtering (if specified) [Savitsky-Golay]
        - Voxel selection across runs using average of tissue probabilities
        - Averaging of timecourses across runs and iterations
        - Perform fitting

    Parameters
    ----------
    -b <bids_dir>       path where 'func' and 'anat' folders live. Can be used instead of individually specifying '-f' and '-a'
    -f <func_dir>       path to where the functional files live. We'll search for the "acq-3DEPI" 
                        tag. If multiple files are found, we'll run preprocessing for them all.
    -a <anat_dir>       path to where the the anatomical slices are. This can be used for aCompCor                       
    -o <output_dir>     output directory + basename for output; some stuff about model type, stage, and parameter type will be appended
    -l <log_dir>        directory that contains the "Screenshot"-directory to create the design matrix
    -r <run ID>         run identifier in case we don't want to average across runs and fit the pRFs for individual runs
    -t <trafo>          transformation file to warp the segmentations from ses-1 to current session
    -i <iterations>     the experiment consists of one set of 8 sweeps (1 iteration) that can be repeated multiple times. We can average these iterations by specifying how many iterations we used. By default, the number of iterations will be read from the log-file from ExpTools, which will always be 1 given that we use the directory that has the screenshots. For that, we only need 1 iterations. With this extra flag you can specify how many iterations to consider.
    -m <model>          Model type to use, e.g., "gauss" or "norm". Defaults to 'norm'.
    --ses_trafo         transformation mapping ses-1 to the closest partial anatomy prior to the corresponding slice of the line-scanning acquisition
    --filter_pca        cutoff frequency for highpass filtering of PCA components. This procedure can ensure that task-related frequencies in the PCA-components do not get removed
    --rsq               r2-threshold to use during pRF-fitting. Parameters of voxels < threshold will be set to 0
    -v                  turn on verbose
    -g                  run model fitter with gray matter voxels only (based on the average tissue probabilities across runs)
    -n                  don't do lowpass filtering on the timecourses
    -q                  do quality control, not full fitting; plots will be stored in the 'anat' folder.

    Returns
    ----------
    pRF-fitting results in the form of *npy-files (including parameters & predictions)

    Example
    ----------
    >>> # set some variables
    >>> sub=007
    >>> ses=2
    >>> iters=2
    >>> filter_pca=0.18
    >>> bids_dir=${DIR_DATA_HOME}/sub-${sub}/ses-${ses}
    >>> out_dir=${DIR_DATA_DERIV}/prf/sub-${sub}/ses-${ses}/sub-${sub}_ses-${ses}
    >>> log_dir=/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/lineprf/logs/sub-${sub}_ses-${ses}_task-pRF_run-imgs
    >>> ses_trafo=${DIR_DATA_DERIV}/pycortex/sub-${sub}/transforms/sub-${sub}_from-ses1_to-ses${ses}_rec-motion1_desc-genaff.mat
    >>> #
    >>> # run locally, filter PCA-components and verbose
    >>> python line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --filter_pca ${filter_pca} -v 
    >>> #
    >>> # run locally, filter PCA-components, verbose, fit only GM-voxels
    >>> python line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --filter_pca ${filter_pca} -v -g
    >>> #
    >>> # run QA-only with verbose
    >>> python line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --filter_pca ${filter_pca} -v -q
    >>> #
    >>> # submit to cluster
    >>> qsub -N prf_${sub} -pe smp 1 -wd /data1/projects/MicroFunc/Jurjen/programs/project_repos/pRFline/logs line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --filter_pca ${filter_pca} -v -g
    """

    func_dir    = None
    anat_dir    = None
    bids_dir    = None
    run_id      = None
    output_dir  = None
    log_dir     = None
    verbose     = False
    model       = "norm"
    ses_trafo   = None
    run_trafos  = None
    n_iter      = None
    filter_pca  = 0.2
    rsq_thresh  = 0.05
    lowpass     = True
    qa          = False
    gm_only     = False

    try:
        opts = getopt.getopt(argv,"nqgvh:a:b:f:d:r:o:l:i:",["func_dir=", "bids_dir=", "n_iter=", "ses_trafo=", "anat_dir=", "output_dir=", "log_dir=", "filter_pca=", "rsq=", "run="])[0]
    except getopt.GetoptError:
        print("ERROR while handling arguments.. Did you specify an 'illegal' argument..?")
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts: 
        if opt == '-h':
            print(main.__doc__)
            sys.exit()
        elif opt in ("-b", "--bids_dir"):
            bids_dir = arg            
        elif opt in ("-f", "--func_dir"):
            func_dir = arg
        elif opt in ("-a", "--anat_dir"):
            anat_dir = arg
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
        elif opt in ("--filter_pca"):
            filter_pca = float(arg)
        elif opt in ("--rsq"):
            rsq_thresh = arg
        elif opt in ("-n"):
            lowpass = False                    
        elif opt in ("-v"):
            verbose = True
        elif opt in ("-g"):
            gm_only = True
        elif opt in ("-q"):
            qa = True                 
    
    if len(argv) == 0:
        print(main.__doc__)
        sys.exit()

    #---------------------------------------------------------------------------------------
    # Error handling of mandatory arguments
    if bids_dir == None:
        if func_dir == None:
            raise ValueError("Please specify the path to the functional files")
    else:
        func_dir = opj(bids_dir, 'func')
        anat_dir = opj(bids_dir, 'anat')

    if output_dir == None:
        raise ValueError("Please specify an output directory (e.g., derivatives/prf/<subject>/<ses-)")

    if log_dir == None:
        raise ValueError("Please specify a log-directory with the Screenshot-directory")

    #---------------------------------------------------------------------------------------
    # check if we should do single runs or all runs at once for the output name of aCompCor figure
    if run_id != None and run_id != "all":
        func_search = ["task-pRF", "bold.mat", f"run-{run_id}"]
        anat_search = ["acq-1slice", ".nii.gz", f"run-{run_id}"]
    else:
        func_search = ["task-pRF", "bold.mat"]
        anat_search = ["acq-1slice", ".nii.gz"]

    func_files = get_file_from_substring(func_search, func_dir)
    ref_slices = get_file_from_substring(anat_search, anat_dir)

    #---------------------------------------------------------------------------------------
    # read some settings from the log-file
    log_file = get_file_from_substring(".yml", log_dir)
    with open(log_file, 'r', encoding='utf8') as f_in:
        settings = yaml.safe_load(f_in)

    # reconstruct design in LineExps/lineprf/session.py
    design          = settings['design']
    bsl_trials      = int(design.get('start_duration')/design.get('stim_duration'))
    sweep_trials    = int(design.get('bar_steps')*2 + (design.get('inter_sweep_blank')//design.get('stim_duration')))
    rest_trials     = int(design.get('blank_duration')//design.get('stim_duration'))
    block_trials    = int(sweep_trials*2 + rest_trials)
    part_trials     = int(block_trials*2 + rest_trials*2)
    iter_trials     = int(part_trials*design.get('stim_repetitions'))
    t_r             = settings['design'].get('repetition_time')
    iter_duration   = iter_trials*design.get('stim_duration')
    
    # check if we got n_iter as argument:
    if n_iter == None:
        n_iter = settings['design'].get('stim_repetitions')
    else:
        n_iter = int(n_iter)

    # check if we should skip the baseline; should be skipped if n_iter > 1 because of weird artifacts of averaging
    if n_iter > 1:
        strip_baseline = True
    else:
        strip_baseline = False

    #---------------------------------------------------------------------------------------
    # check if we got anatomies + trafo files
    if anat_dir != None:
        if ses_trafo == None:
            raise ValueError("Need a transformation file mapping ses-1 to current session")

        if isinstance(func_files, list):
            try:
                run_trafos = get_file_from_substring(".txt", anat_dir)
            except:
                raise ValueError(f"Received multiple functional files and an anatomical folder, but no run-to-run registration files. Manually align the slices and save them as txt-files in {anat_dir}")
    
    #---------------------------------------------------------------------------------------
    # set output basename
    if run_id != None:
        output_base_prf = os.path.basename(output_dir)+f"_run-{run_id}"
    else:
        output_base_prf = os.path.basename(output_dir)

    #---------------------------------------------------------------------------------------
    # initiate model fitting
    model_fit = fitting.FitLines(func_files=func_files,
                                 TR=t_r,
                                 low_pass=lowpass,
                                 window_size=19,
                                 log_dir=log_dir,
                                 stage='grid+iter',
                                 model=model,
                                 baseline_duration=design.get('start_duration'),
                                 iter_duration=iter_duration,
                                 n_iterations=n_iter,
                                 verbose=verbose,
                                 strip_baseline=strip_baseline,
                                 acompcor=True,
                                 ref_slice=ref_slices,
                                 filter_pca=filter_pca,
                                 rsq_threshold=rsq_thresh,
                                 ses1_2_ls=ses_trafo,
                                 run_2_run=run_trafos,
                                 output_dir=os.path.dirname(output_dir),
                                 output_base=output_base_prf,
                                 save_as=opj(anat_dir, os.path.basename(output_dir)),
                                 voxel_cutoff=300,
                                 ribbon=gm_only)

    # fit
    if not qa:
        model_fit.fit()

if __name__ == "__main__":
    main(sys.argv[1:])
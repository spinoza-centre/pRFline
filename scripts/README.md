# Scripts

## Partial FOV

As extra confirmation of our planning, we acquire a 3D-EPI partial FOV run with the same experiment as the line experiment (https://github.com/gjheij/LineExps/lineprf). Because `fMRIprep` does not work well with this level of partial FOV, I've had to adapt the pipeline slighty. I now do the following steps (see https://github.com/spinoza-centre/pRFline/tree/main/data for execution):
- Motion/distortion correction (`call_topup`)
- Apply inverse of `from-ses1_to-ses2` matrix to create a `temporary`-space `call_antsapplytransforms` close to `T1w`-space
- Refine this `bold-to-T1w` registration with `fMRIPrep`'s implementation of `bbregister` (`call_bbregwf`)
- Project to volumetric `FSNative`-space via refined registration to `T1w` (`call_antsapplytransforms`)

We can then use `partial_fit.py` to fit a pRF-model to this data (you can check with FreeView that we have reasonable overlap with the structural image despite the limited FOV). Internally, this calls on `pRFmodelFitting` in https://github.com/gjheij/linescanning/blob/main/linescanning/prf.py. The call is as follows:

```bash
# set subject/session
subID=002
sesID=2

# set path
bids_dir=${DIR_DATA_DERIV}/fmriprep
log_dir=${DIR_DATA_SOURCE}/sub-${subID}/ses-${sesID}/sub-${subID}_ses-${sesID}_task-pRF_run-imgs

# submit to cluster or run locally with python
# job="python"
job="qsub -N pfov_${subID} -pe smp 5 -wd /data1/projects/MicroFunc/Jurjen/programs/project_repos/pRFline/logs"

${job} partial_fit.py -s ${subID} -n ${sesID} -b ${bids_dir} -l ${log_dir} -v --fsnative # fit with fsnative
```

## Line

For the line-experiments, we can run `line_fit.py`, which internally preprocessed the functional runs (high/low-pass filtering), and averages over runs and design iterations (2x runs and 3x iterations per run). It also strips the run from it's baseline, because of wonky averaging and selects the ribbon-voxels from the dataframe to limit the demand on resources. We create a separate design matrix from the same screenshots because of different repetition times (`1.111s` vs `0.105`s). First, we need some registration matrices for:

- Mapping `ses-1` to the image closest to the line-scanning acquisition (generally, this will be the `rec-motion1` image). Based on the subject & session IDs, it selects `ses-1` data from `DIR_DATA_DERIV/pymp2rage` and the `ses-${sesID}` data (low-res anatomical image) from `$DIR_DATA_HOME`.

  ```bash
  # subject/session information
  subID=009
  sesID=2

  # shortcuts
  base_path=sub-${subID}/ses-${sesID}
  base=sub-${subID}_ses-${sesID}

  # cmd
  call_ses1_to_motion1 sub-${subID} ${sesID}
  ```

- Registrations mapping each indivual anatomical slice to the slice closest to the partial FOV anatomical data (again, usually this is the `rec-motion1` image), so we can project all the segmentations to each run as accurately as possible. For this, we need to run ITK-Snap a bunch of times and save the files as `from-run1_to-runX.txt`, in `$DIR_DATA_HOME/sub-${subID}/ses-${sesID}/anat`:

  ```bash
  n_runs=3
  subj_dir=$DIR_DATA_HOME/${base_path}/anat
  mov=${subj_dir}/${base}_acq-1slice_run-1_T1w.nii.gz
  for run in `seq 1 ${n_runs}`; do

    if [[ ${run} -eq 1 ]]; then
      call_createident ${subj_dir}/from-run1_to-run${run}.txt
    else
      ref=${subj_dir}/${base}_acq-1slice_run-${run}_T1w.nii.gz

      # open ITK-Snap > ctrl+R  (or 'cmd+R' on mac) > 'manual' > align
      # press the floppy button on the bottom right, save as:
      echo "save image as ${subj_dir}/from-run1_to-run${run}.txt"
      itksnap -g ${ref} -o ${mov}
    fi
  done
  ```
 The call is as follows:

```bash

# directories
bids_dir=${DIR_DATA_HOME}/${base_path}
out_dir=${DIR_DATA_DERIV}/prf/${base_path}/${base}_task-pRF
log_dir=${DIR_DATA_SOURCE}/${base_path}/${base}_task-pRF_run-imgs
ses_trafo=${DIR_DATA_DERIV}/pycortex/sub-${subID}/transforms/sub-${subID}_from-ses1_to-ses${sesID}_rec-motion1_desc-genaff.mat

# submit to cluster or run locally with python
# job="python"
job="qsub -N line_${subID} -pe smp 5 -wd /data1/projects/MicroFunc/Jurjen/programs/project_repos/pRFline/logs"

# please run with --qa first to potentially exclude runs based on the heuristics in the report
python line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --verbose --qa
```
Then, check the subject's html-file to check for heavy motion (e.g., coughing). Exclude those runs using:

### Whole-line pRF-fitting
The most vanilla version of pRF-fitting is simply fitting across the entire line. That is done as follows:
```bash
${job} line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --verbose --exclude 4 # excludes run-4

# or
${job} line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --verbose --exclude 2,3 # excludes run-2/3
```

### pRF-fitting on ribbon (GM-voxels around target vertex)
We can also fit only on the ribbon voxels. In that case, we need to go to FSLeyes/ITK-Snap and open our CRUISE-segmentation image and the 'beam' image. This way, we can verify which voxels belong to the ribbon of interest. Here's where stuff gets more exotic. We can first estimate the pRFs over the average across the ribbon (which would imitate whole-brain fitting). We can then fix the x/y parameters, and estimate the parameters across the command. We can do that with the command:

```bash
${job} line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --verbose --ribbon 356,363 --fix 0,1 --feed_avg --no_grid --exclude 1

# --exclude 1       = exclude run-1 > check the report html that was created earlier
# --ribbon 356,363  = range of ribbon voxels
# --feed_avg        = first fit on the average across ribbon, then feed those parameters into individual fit
# --fix 0,1         = fix X/Y parameters from average fit
# --no_grid         = don't save grid-parameters (save clogging up directory)
```

### HRF fitting
We can also add the `--hrf` flag to fit the HRF during pRF-fitting. If you've ran standard fit already, you can run the same command with the `--hrf` flag, and the old parameters will be used as starting point. So it doesn't start fitting from scratch with _more_ parameters.
```bash
${job} line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --verbose --hrf
```

If you quickly want to test if stuff works without making a new `fMRIPrep`-like report or `aCompCor`, you can do:
```bash
${job} line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --verbose --hrf --no_acompcor --no_report
```

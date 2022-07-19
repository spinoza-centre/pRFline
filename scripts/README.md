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

For the line-experiments, we can run `line_fit.py`, which internally preprocessed the functional runs (high/low-pass filtering), and averages over runs and design iterations (2x runs and 3x iterations per run). It also strips the run from it's baseline, because of wonky averaging and selects the ribbon-voxels from the dataframe to limit the demand on resources. We create a separate design matrix from the same screenshots because of different repetition times (`1.111s` vs `0.105`s). The call is as follows:

```bash
# subject/session information
subID="007"
sesID=2

# defaults
iters=2
filter_pca=0.18

# directories
bids_dir=${DIR_DATA_HOME}/sub-${subID}/ses-${sesID}
out_dir=${DIR_DATA_DERIV}/prf/sub-${subID}/ses-${sesID}/sub-${subID}_ses-${sesID}_task-pRF
log_dir=${DIR_DATA_SOURCE}/sub-${subID}/ses-${sesID}/sub-${subID}_ses-${sesID}_task-pRF_run-imgs
ses_trafo=${DIR_DATA_DERIV}/pycortex/sub-${subID}/transforms/sub-${subID}_from-ses1_to-ses${sesID}_rec-motion1_desc-genaff.mat

# submit to cluster or run locally with python
# job="python"
job="qsub -N line_${subID} -pe smp 5 -wd /data1/projects/MicroFunc/Jurjen/programs/project_repos/pRFline/logs"

# actual call (make sure correct environment is loaded!)
${job} line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --filter_pca ${filter_pca} -v
```

We can also add the `--hrf` flag to fit the HRF during pRF-fitting. If you've ran standard fit already, you can run the same command with the `--hrf` flag, and the old parameters will be used as starting point. So it doesn't start fitting from scratch with _more_ parameters.
```bash
${job} line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --filter_pca ${filter_pca} --verbose --hrf
```

If you quickly want to test if stuff works without making a new `fMRIPrep`-like report or `aCompCor`, you can do:
```bash
${job} line_fit.py -b ${bids_dir} -o ${out_dir} -l ${log_dir} --ses_trafo ${ses_trafo} -i ${iters} --filter_pca ${filter_pca} --verbose --hrf --no_acompcor --no_report
```

# Scripts

## Partial FOV

As extra confirmation of our planning, we acquire a 3D-EPI partial FOV run with the same experiment as the line experiment (https://github.com/gjheij/LineExps/lineprf). Because `fMRIprep` is a slight overkill for the current purposes (aside from difficulties with registering extremely partial FOVs), I run the script below:

```bash
python partial_preprocess.py -s sub-003 -f /data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -o /data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -n 3
```

This performs the following steps:

* Motion correction
* Registration to ses-1 (orig.mgz)
* Projection to `fsnative`

We can then use `partial_fit.py` to fit a pRF-model to this data (you can check with FreeView that we have reasonable overlap with the structural image despite the limited FOV). Internally, this calls on https://github.com/gjheij/linescanning/blob/main/linescanning/prf.py#L1088. The call is as follows:

```bash
python partial_fit.py -s sub-003 -f /data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -o /data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -l /data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/prf_design/sub-003_ses-0_task-pRF_run-0 -v
```

The `-v` signifies we want to print some information to the terminal. We can also submit this job to a cluster:

```bash
qsub -N prf_003 -pe smp 1 -wd /data1/projects/MicroFunc/Jurjen/programs/project_repos/pRFline/logs partial_fit.py -s sub-003 -f /data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -o /data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -l /data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/prf_design/sub-003_ses-0_task-pRF_run-0 -v
```

## Line
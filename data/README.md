# Steps for dealing with partial FOV data
## Denoising with NORDIC
Ones has been exported from the scanner, we can convert the data to nifti and make it more or less BIDS-compatible with the following command, which internally will execute [spinoza_scanner2bids](https://github.com/gjheij/linescanning/blob/main/shell/spinoza_scanner2bids):
```bash
subID="008" # sub-008
sesID="2"   # ses-2
master -m 02 -s ${subID} -n ${sesID}
```

This will produce a `func`-folder in `$DIR_DATA_SOURCE/sub-${subID}/ses-${sesID}/func`:
```bash
func/
├── nordic.m # will be created
├── sub-${subID}_ses-${ses_ID}_task-pRF_run-1_acq-3DEPI_bold.json
├── sub-${subID}_ses-${ses_ID}_task-pRF_run-1_acq-3DEPI_bold.nii.gz
├── sub-${subID}_ses-${ses_ID}_task-pRF_run-1_acq-3DEPI_bold_ph.json
├── sub-${subID}_ses-${ses_ID}_task-pRF_run-1_acq-3DEPI_bold_ph.nii.gz
├── sub-${subID}_ses-${ses_ID}_task-pRF_run-1_acq-3DEPI_epi.json
└── sub-${subID}_ses-${ses_ID}_task-pRF_run-1_acq-3DEPI_epi.nii.gz
```

These files we can use to perform `NORDIC`-denoising. The [call_nordic](https://github.com/gjheij/linescanning/blob/main/bin/call_nordic) command is pretty simple:

```
---------------------------------------------------------------------------------------------------
call_nordic
Applies the NORDIC algorithm to whole-brain data. Requires the magnitude image, the phase image, and 
an output name
Args:
  -m        use magnitude only
  <mag>     magnitude image
  <phase>   phase image
  <output>  nordic'ed output
Usage:
  call_nordic [-m] <mag file> <phase file> <nordic output>
Example:
  call_nordic func_mag.nii.gz func_phase.nii.gz func_nordic.nii.gz      # mag + phase
  call_nordic -m func_mag.nii.gz func_phase.nii.gz func_nordic.nii.gz   # mag only
---------------------------------------------------------------------------------------------------
```

So, we'll just run:
```bash
# in the 'func' of sourcedata
mag="sub-${subID}_ses-${ses_ID}_task-pRF_run-1_acq-3DEPI_bold.nii.gz"
phase="sub-${subID}_ses-${ses_ID}_task-pRF_run-1_acq-3DEPI_bold.nii.gz"

# in project root folder
out="${DIR_DATA_HOME}/sub-${subID}/ses-${sesID}/sub-${subID}_ses-${ses_ID}_task-pRF_run-1_acq-3DEPI_bold.nii.gz"

# cmd
call_nordic ${mag} ${phase} ${out}
```

# Fix json-sidecars
For [fMRIPrep](https://fmriprep.org/en/stable/index.html) to deal with distortion correction properly, we need the `PhaseEncodingDirection` and `IntendedFor` fields in the `json`-files of the `epi`-files and the `PhaseEncodingDirection` in the `bold`-files. Sometimes `dcm2niix` will insert `PhaseEncodingAxis` instead in [situations where the axis is known but the polarity is not](https://neurostars.org/t/bids-func-metadata-phaseencodingaxis-or-phaseencodingdirection/4558/2). Therefore, we need to manually check the json-files and add the fields. Open the `*_epi.json`-files in an editor and add at the bottom (check [here](https://github.com/nipreps/fmriprep/issues/1493) what you need to enter in the `PhaseEncodingDirection` field). Moreover, you have to change the `EstimatedEffectiveEchoSpacing` and `EstimatedTotalReadoutTime` to `EffectiveEchoSpacing` and `TotalReadoutTime`.

Total readout time is required for TOPUP and can be calculated as:
```python
# Check PAR-file for water-fat shift
WFS = 38.445 
#magnetic field strength * water fat difference in ppm * gyromagnetic hydrogen ratio
WFS_hz = 7 * 3.35 * 42.576

# total readout time
TRT = WFS/WFS_hz

# EffectiveEchoSpacing
EES = TRT / EPI_FACTOR+1 # see PAR-file for epi factor
```

Then add this information to the files in `fmap/*_epi.nii.gz` and `func/*_bold.nii.gz`:
```python
bold_file = "ses-2/func/sub-001_ses-2_task-pRF_run-1_bold.nii.gz"

# decide phase encoding direction for EPI and BOLD files (must be opposite):
pedir = {'i': 'Left-Right', 'i-': 'Right-Left', 'j-': 'Anterior-Posterior', 'j': 'Posterior-Anterior'}
```

EPI-FILE
```json
	"PhaseEncodingDirection": "i", // epi_pe
	"IntendedFor": "ses-2/func/sub-008_ses-2_task-pRF_run-1_acq-3DEPI_bold.nii.gz", // bold_file
	"TotalReadoutTime": 0.055007, // TRT
	"EffectiveEchoSpacing": 0.00144756 // EES
}

```

BOLD-FILE
```json
	"TotalReadoutTime": 0.055007, // TRT
	"PhaseEncodingDirection": "i-" // bold_pe
}

```

Now we're done with the `epi`-files so we can copy them to the corresponding `fmap`-folder:
```bash
mkdir -p ${DIR_DATA_HOME}/${sub_id}/ses-${ses_id}/fmap
cp *_epi* ${DIR_DATA_HOME}/${sub_id}/ses-${ses_id}/fmap
```

# Partial preprocessing with fMRIPrep
I have yet to encounter a partial FOV dataset such as this to be fully compatible with [fMRIPrep](https://fmriprep.org/en/stable/index.html). Most often, the brain-masking will fail which results in problems with `aCompCor`. Additionally, registration is a pain. Because I already have some sort of registration matrix (the one mapping `FreeSurfer` (*ses-1*) to *ses-2*, I modified the [fMRIPrep](https://fmriprep.org/en/stable/index.html)-workflow to only do `motion correction` and `distortion correction`).

This is all wrapped in the `call_topup`-command:
```
---------------------------------------------------------------------------------------------------
call_topup

This script runs only the initializing nodes from fMRIPrep. E.g., is will do all the header stuff, 
validation, but most most importantly, motion correction and topup. Under the hood, it utilizes the
functions in 'linescanning.fmriprep', which are literally the workflows from fMRIPrep, but with all
other irrelevant stuff commented out (it's still there just in case). 

Input needs to be the full project root directory so the correct fieldmaps can be found. Using the 
fmriprep_config?.json files you can select particular subsets of the entire dataset. E.g., I only 
use this script for my limited FOV data, as the brainmasking fails on that data causing aCompCor to
fail. Therefore, I just need motion-corrected, topup data for a very select part of the dataset. 

Parameters
----------
    -s  subject ID
    -i  subject-directory containing the files that need to be run through Topup.
    -o  output directory containing the topup'ed data. Easiest is to give the fMRIPrep folder, it
        will be formatted accordingly automatically
    -w  working directory where fMRIPrep's intermediate files are stored; default is some folder 
        in /tmp
    -b  bids filter file (maps to --bids-filter-file from fMRIPrep); allows you to select parti-
        cular files
                        
Example
----------
>>> call_topup -s 001 -i DIR_DATA_HOME -o DIR_DATA_DERIV/fmriprep -w DIR_DATA_SOURCE/sub-001/ses-2
>>> call_topup -s 001 -i DIR_DATA_HOME -o DIR_DATA_DERIV/fmriprep -b misc/fmriprep_config1.json

---------------------------------------------------------------------------------------------------
```
Note that this is a bash script, while the workflows are implemented in python. This is because when running a python version of this script in the terminal, nodes will mix up during `fmap_wf.resample`, causing the process to crash. This does not happen when you run it interactively in python, so this bash script sends a block of code to python:

```bash
# this is the actual code
PYTHON_CODE=$(cat <<END
from linescanning import fmriprep
wf = fmriprep.init_single_subject_wf("${subject}", 
                                     bids_dir="${inputdir}", 
                                     fmriprep_dir="${outputdir}", 
                                     bids_filters="${bids_filters}", 
                                     workdir="${workdir}")
wf.run()                                     
END
)

# run the code
res="$(python -c "$PYTHON_CODE")"
```

It might not be the cleanest operation, but it's pretty cool and it works. So I'll take it.

We can construct our command as follows:
```bash
call_topup -s ${subID} -i ${DIR_DATA_HOME} -b ${DIR_SCRIPTS}/misc/fmriprep_config1.json -w ${DIR_DATA_SOURCE}/sub-${subID}/ses-2
```

## Project to ses-1 surface
This partial FOV acquisition serves as an additional confirmation of the location of the vertex. To do so, we need to project the data to the same surface on which the pRF-estimates from `ses-1` are based. We'll do this in two steps:

1) Get matrix mapping `ses-1` to `ses-2` with [call_ses1_to_ses](https://github.com/gjheij/linescanning/blob/main/bin/call_ses1_to_ses)
2) Get matrix mapping `bold` to `T1w` with `call_bbregwf`
3) Apply `bold-to-T1w` and `T1w-to-fsnative` in one step
4) Project this to the surface with [call_vol2fsaverage](https://github.com/gjheij/linescanning/blob/main/bin/call_vol2fsaverage)

### Matrix mapping `ses-1` to `ses-2`
```bash
# set sub/ses IDs
subID=008
sesID=2

# create file
call_ses1_to_ses sub-${subID} ${sesID}
matrix1=${DIR_DATA_DERIV}/pycortex/sub-${subID}/transforms/sub-${subID}_from-ses1_to-ses${sesID}_desc-genaff.mat
```

### Matrix mapping bold of `ses-2` to T1w of `ses-1`
```bash
# get bold file
bold=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-${sesID}/func/sub-${subID}_ses-${sesID}_task-pRF_acq-3DEPI_run-1_desc-preproc_bold.nii.gz

# get reference
ref=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-1/anat/sub-${subID}_ses-1_acq-MP2RAGE_desc-preproc_T1w.nii.gz

# set output; call it space-tmp for now as we need to refine with bbregister
out=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-${sesID}/func/sub-${subID}_ses-${sesID}_task-pRF_acq-3DEPI_run-1_space-tmp_desc-preproc_bold.nii.gz
json=$(dirname ${out})/$(basename ${out} .nii.gz).json

# make json file
if [ -f ${json} ]; then
    rm ${json}
fi

(
echo "{"
echo "  \"RepetitionTime\": 1.11149,"
echo "  \"SkullStripped\": false,"
echo "  \"SliceTimingCorrected\": false,"
echo "  \"MatrixApplied\": \"${matrix1}\","
echo "  \"SourceFile\": \"${bold}\","
echo "  \"Description\": \"temporary space (to be refined with BBR) before projection to FSNative\""
echo "}" 
) >> ${json}

# we need to invert matrix to go to tmp-space > bbregister will fail is the partial FOV itself is inserted
call_antsapplytransforms -i 1 ${ref} ${bold} ${out} ${matrix1}

# make boldref
boldref1=$(dirname ${out})/$(basename ${out} desc-preproc_bold.nii.gz)boldref.nii.gz
fslmaths ${out} -Tmean ${boldref1}
```

This registration is not perfect, so we can refine the registration with `bbregister` from here by taking another workflow from fMRIPrep called `bold_reg_wf`, and we'll input the motion/distortion corrected registered file (`boldref`) as input. The output consists of the directory + basename:
```bash
workdir=${DIR_DATA_SOURCE}/sub-${subID}/ses-2/single_subject_${subID}_wf/func_preproc_ses_${sesID}_task_pRF_run_1_acq_3DEPI_wf # store with other workflow outputs
outdir=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-${sesID}/func/sub-${subID}_ses-${sesID}_task-pRF_acq-3DEPI_run-1 # 'from-{bold|T1w}_to-{bold|T1w}_mode-image_xfm.txt will be appended
call_bbregwf -s ${subID} -b ${boldref1} -w ${workdir} -o ${outdir}
```

## Combine `from-tmp_to-T1w` and `from-T1w_to-fsnative` in one step:
```bash
# set old output to new input
bold=${out}

# set new output
out=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-${sesID}/func/sub-${subID}_ses-${sesID}_task-pRF_acq-3DEPI_run-1_space-fsnative_desc-preproc_bold.nii.gz

# define json file
json=$(dirname ${out})/$(basename ${out} .nii.gz).json

# get the anatomical from ses-1
anat=${DIR_DATA_DERIV}/freesurfer/sub-${subID}/mri/orig.nii.gz

# from bold-to-T1w
matrix2=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-${sesID}/func/sub-${subID}_ses-${sesID}_task-pRF_acq-3DEPI_run-1_from-bold_to-T1w_mode-image_xfm.txt

# get the T1w-to-fsnative matrix
matrix3=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-1/anat/sub-${subID}_ses-1_acq-MP2RAGE_from-T1w_to-fsnative_mode-image_xfm.txt

# make json file
if [ -f ${json} ]; then
    rm ${json}
fi

(
echo "{"
echo "  \"RepetitionTime\": 1.11149,"
echo "  \"SkullStripped\": false,"
echo "  \"SliceTimingCorrected\": false,"
echo "  \"MatrixApplied\": [\"${matrix2}\", \"${matrix3}\"],"
echo "  \"SourceFile\": \"${bold}\","
echo "  \"Description\": \"volumetric FSnative space\""
echo "}" 
) >> ${json}

# apply the matrix to space-tmp to get space-fsnative > "0 0" means invert neither of the matrices
call_antsapplytransforms -i "0 0" ${anat} ${bold} ${out} "${matrix2} ${matrix3}"

# make boldref
boldref2=$(dirname ${out})/$(basename ${out} desc-preproc_bold.nii.gz)boldref.nii.gz
fslmaths ${out} -Tmean ${boldref2}
```

### Project to surface
Now that we have the bold data in `FSNative`-space, our final task is to project it to the surface. This is actually the easiest step, we just have to run [call_vol2fsaverage](https://github.com/gjheij/linescanning/blob/main/bin/call_vol2fsaverage), and the required intermediate files are produced:

```bash
outdir=$(dirname ${out})

# 'space-{fsnative|fsaverage}' will be inserted inbetween $prefix and $suffix
prefix=sub-${subID}_ses-${sesID}_task-pRF_acq-3DEPI_run-1
suffix=bold.func

# build command
call_vol2fsaverage -t -o ${outdir} -p ${prefix} sub-${subID} ${out} ${suffix}
```

Aside from `gii`-files, [call_vol2fsaverage](https://github.com/gjheij/linescanning/blob/main/bin/call_vol2fsaverage) will also produce a numpy array in which the `gii`'s from the left and right hemisphere (for both `fsnative` and `fsaverage`) are stacked onto one another (*left* first). This numpy array can be directly used in conjunction with [the dataset class](https://github.com/gjheij/linescanning/blob/main/linescanning/dataset.py#L1657)

## Brain extract 3D-EPI with brain-mask from T1w
The whole reason we do this exercise is because fMRIPrep fails on the brain mask. Technically we have all the ingredients to brainmask the 3D-EPI data: a warp mapping bold to T1w, and a good brainmask in T1w-space. We can therefore apply the warp to the mask to create a bold mask.

```bash
# set the output to something fMRIPreppy
bold_mask=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-${sesID}/func/sub-${subID}_ses-${sesID}_task-pRF_acq-3DEPI_run-1_desc-brain_mask.nii.gz

# mask
t1_mask=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-1/anat/sub-${subID}_ses-1_acq-MP2RAGE_desc-brain_mask.nii.gz

# apply
call_antsapplytransforms -v -i "0 1" -t gen ${boldref1} ${t1_mask} ${bold_mask} "${matrix1} ${matrix2}"

# dilate slightly
call_dilate ${bold_mask} ${bold_mask} 2

# apply mask
bold=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-${sesID}/func/sub-${subID}_ses-${sesID}_task-pRF_acq-3DEPI_run-1_desc-preproc_bold.nii.gz
out=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-${sesID}/func/sub-${subID}_ses-${sesID}_task-pRF_acq-3DEPI_run-1_desc-masked_bold.nii.gz

fslmaths ${bold} -mas ${bold_mask} ${out}

# json file
json=$(dirname ${out})/$(basename ${out} .nii.gz).json
if [ -f ${json} ]; then
    rm ${json}
fi

(
echo "{"
echo "  \"RepetitionTime\": 1.11149,"
echo "  \"SkullStripped\": true,"
echo "  \"SliceTimingCorrected\": false,"
echo "  \"MaskSource\": \"${bold_mask}\","
echo "  \"SourceFile\": \"${bold}\","
echo "  \"Description\": \"motion/distortion correct + brain extraction\""
echo "}" 
) >> ${json}

# boldref
boldref3=$(dirname ${out})/$(basename ${out} .nii.gz)ref.nii.gz
fslmaths ${out} -Tmean ${boldref3}
```
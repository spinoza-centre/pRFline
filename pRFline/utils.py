from linescanning import utils, prf
import os
from nipype.interfaces import fsl, ants, freesurfer
opj = os.path.join

def split_params_file(file):

    comp_list = fname.split('_')
    comps = {}
    
    ids = ['sub', 'ses', 'run', 'model', 'stage']
    for el in comp_list:
        for i in ids:
            if i in el:
                comp = el.split('-')[-1]
                if i == "run":
                    comp = int(comp)

                comps[i] = comp

    if len(comps) != 0:
        return comps
    else:
        print(f"Could not find any element of {ids} in {fname}")
    
def mcflirt(func):

    print(f"running mcflirt on {func}")
    mcflt = fsl.MCFLIRT()
    mcflt.inputs.in_file    = func
    mcflt.inputs.cost       = 'mutualinfo'
    print(mcflt.cmdline)
    res = mcflt.run()

    return res

def nordic(func, phase=None, output=None):
    print(f"running nordic on {func}")

    if output != None:
        output = output
    else:
        output=opj(os.path.dirname(func), 'nordic_bold.nii.gz')

    if not os.path.exists(output):
        cmd=f"call_nordic {func} {phase} {output}"
        print(cmd)
        os.system(cmd)
    
    return output

def ants_apply(moving, fixed, output, transform, invert=True):
    print("running antsApplyTransforms")

    at = ants.ApplyTransforms()
    at.inputs.dimension                 = 3
    at.inputs.input_image               = moving
    at.inputs.reference_image           = fixed
    at.inputs.output_image              = output
    at.inputs.interpolation             = 'Linear'
    at.inputs.transforms                = transform
    at.inputs.invert_transform_flags    = invert
    at.inputs.input_image_type          = 3
    at.inputs.float                     = True

    print(at.cmdline)
    at.run()

    return at

def mri_vol2surf(sval, tval, subject=None, reg_file=None, hemi=None):
    print(f"running mri_vol2surf on {sval}")

    # create identity .dat file
    if reg_file == None:
        if subject == None:
            subject = "sub-001"

        reg_file = opj(os.path.dirname(sval), 'ident.dat')
        cmd = f"call_createident -s {subject} {reg_file} {sval}"
        os.system(cmd)

    if not os.path.exists(sval):
        raise FileNotFoundError(f"{sval} does not exist")
    
    # sample
    sampler = freesurfer.SampleToSurface(hemi=hemi)
    sampler.inputs.source_file      = sval
    sampler.inputs.reg_file         = reg_file
    sampler.inputs.sampling_method  = "point"
    sampler.inputs.sampling_range   = 0.5
    sampler.inputs.sampling_units   = "frac"
    sampler.inputs.out_file         = tval
    sampler.inputs.out_type         = 'gii'
    print(sampler.cmdline)
    sampler.run()

    return sampler

def mri_surf2surf(sval, subject=None, target="fsaverage", tval=None, hemi=None):

    sxfm = freesurfer.SurfaceTransform()
    sxfm.inputs.source_file         = sval
    sxfm.inputs.source_subject      = subject
    sxfm.inputs.target_subject      = target
    sxfm.inputs.hemi                = hemi
    sxfm.inputs.output_file         = tval
    sxfm.run() 

def preprocess_func(func, subject=None, phase=None, trafo=None, reference=None, invert=True, outputdir=None):

    if outputdir == None:
        outputdir = os.path.dirname(func)

    comps = utils.split_bids_components(func)
    fbase = f"sub-{comps['sub']}_ses-{comps['ses']}_task-{comps['task']}_run-{comps['run']}_acq-{comps['acq']}"

    # nordic 
    nord_file = opj(outputdir, fbase+"_bold_nordic.nii.gz")
    if phase != None:
        funcs = nordic(func, phase, output=nord_file)
    else:
        funcs = func

    # motion correct
    mcf_file = opj(outputdir, fbase+"_bold_nordic_mcf.nii.gz")
    if not os.path.exists(mcf_file):
        mcflt = mcflirt(funcs)
        mcf_file = mcflt.outputs.out_file

    # apply ses-1 to ses-X so that data is in FS-space
    if trafo != None:
        fname = f"{fbase}_space-fsnative_bold.nii.gz"
        output = opj(outputdir, fname)

        if not os.path.exists(output):
            ap = ants_apply(mcf_file, reference, output, trafo, invert=invert)
            output = ap.inputs.output_image

    # project to fsnative
    files = []
    for hemi in ['lh', 'rh']:
        if hemi == "lh":
            hemi_tag = "hemi-L"
        else:
            hemi_tag = "hemi-R"

        fsnative = opj(outputdir, f"{fbase}_{hemi_tag}_space-fsnative_bold.gii")
        files.append(fsnative)
        mri_vol2surf(output, fsnative, subject=subject, hemi=hemi)

    # combine them in numpy array
    final = opj(outputdir, f"{fbase}_hemi-LR_space-fsnative_bold.npy")

    cmd = f"call_stackgifti {files[0]} {files[1]} {final}"
    os.system(cmd)
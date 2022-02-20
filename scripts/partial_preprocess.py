#!/usr/bin/env python

import os
import sys, getopt
from pRFline import utils
from linescanning.utils import get_file_from_substring
opj = os.path.join

def main(argv):

    """partial_preprocess.py

    Preprocess the partial FOV functional files prior to pRF-fitting. Steps included are: NORDIC, motion
    correction, registration to FSnative, projection to surface, and high pass filtering with DCT-set.

    Parameters
    ----------
    -s <subject>        subject ID as used throughout the pipeline.
    -f <func_dir>       path to where the functional files live. We'll search for the "acq-3DEPI" 
                        tag. If multiple files are found, we'll run preprocessing for them all.
    -d <derivatives>    path pointing to the derivatives folder including 'freesurfer' and 'pycortex'
    -o <output_dir>     output directory; should be project root directory with <subject>/<ses-X>/
                        func.

    Returns
    ----------
    *npy-file containing 2D data that has been NORDIC'ed, motion corrected, projected to the surface, and 
    high pass filtered

    Example
    ----------
    python partial_preprocess.py -s sub-003 -f /mnt/export/data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -o /mnt/export/data1/projects/MicroFunc/Jurjen/projects/hemifield/testing/preprocess_func -n 3
    """

    subject     = None
    session     = None
    func_dir    = None
    derivatives = os.environ.get("DIR_DATA_DERIV")
    output_dir  = None

    try:
        opts = getopt.getopt(argv,"hs:n:f:d:o:",["subject=", "session=", "func_dir=", "derivatives=", "output_dir="])[0]
    except getopt.GetoptError:
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--subject"):
            subject = arg
        elif opt in ("-n", "--session"):
            session = arg            
        elif opt in ("-f", "--func_dir"):
            func_dir = arg            
        elif opt in ("-d", "--derivatives"):
            derivatives = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg

    if len(argv) < 4:
        print("NOT ENOUGH ARGUMENTS SPECIFIED")
        print(main.__doc__)
        sys.exit()

    # fetch func files, phase files, transformation file mapping ses-1 to ses-X, and reference image (orig.nii.gzs)
    func_files  = get_file_from_substring(["acq-3DEPI", "bold.nii.gz"], func_dir)
    phase_files = get_file_from_substring(["acq-3DEPI", "bold_ph.nii.gz"], func_dir)
    trafo       = get_file_from_substring(f"from-fs_to-ses{session}_", opj(derivatives, 'pycortex', subject, 'transforms'))
    ref_img     = opj(derivatives, 'freesurfer', subject, 'mri', 'orig.nii.gz')

    # loop through files if input is list
    if isinstance(func_files, list):
        for func in func_files:
            print(f"Preprocessing {func}")
            utils.preprocess_func(func, 
                                  subject=subject, 
                                  phase=phase_files, 
                                  trafo=trafo, 
                                  reference=ref_img, 
                                  outputdir=output_dir)
    # input is single file
    elif isinstance(func_files, str):
        print(f"Preprocessing {func_files}")
        utils.preprocess_func(func_files, 
                              subject=subject, 
                              phase=phase_files, 
                              trafo=trafo, 
                              reference=ref_img, 
                              outputdir=output_dir)
    # unknown input
    else:
        raise ValueError(f"Unrecognized input type for {func_files}. Must be list of strings or single string")

    print("Done")

if __name__ == "__main__":
    main(sys.argv[1:])
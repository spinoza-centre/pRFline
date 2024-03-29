#!/usr/bin/env python

import os
import sys
import getopt
import pRFline
from pRFline.utils import SubjectsDict
from linescanning import utils, pycortex, optimal
import time
import numpy as np
import pandas as pd
opj = os.path.join
opd = os.path.dirname

def main(argv):

    """slice_on_surf.py

Project the nominal line image (or any other slice image for that matter) to the surface.

Parameters
----------
    -i|--img            image type (e.g., 'ribbon', 'slice', or 'beam'). Default = 'beam'
    -s|--subject        process specific subject. Default = "all"
    -v|--no_verbose     turn off verbose (best to have verbose on by default)
    -t|--interp         interpolation method. Use '-t gen' for masks, '-t lin' for scalar images. See also call_antsapplytransforms for more options
    --recon             project recon image (default is beam-image)
    --webshow           open pycortex in browser to create images (also saves them). Default is False
    --no_target         don't add the target to the vertex object (default is True)
    --plot              make figure of all subjects (default is False)
    --no_verbose        turn off verbose (default is verbose=True)
    --stats             get curvature values from ribbon vertices

Returns
----------
Opens a webshow from pycortex to save out the figure, as well as pandas dataframe with the distance for each subject

Example
----------
>>> ./slice_on_surf.py
>>> ./slice_on_surf.py -s sub-001
    """

    verbose = True
    subject = "all"
    full_plot = False
    img_type = "beam"
    image = None
    interp = "gen"
    warp = None
    add_target = True
    output_dir = None
    overwrite = False
    skip_img = False
    warp = None
    webshow = False
    curve_stats = False

    try:
        opts = getopt.getopt(argv,"hm:v:k:i:o:t:w:s:",["img=", "help", "subject=", "no_verbose", "webshow","plot", "recon", "in=", "interp=", "warp=", "out=", "ow", "overwrite", "skip_img", "stats"])[0]
    except getopt.GetoptError:
        print("ERROR while handling arguments.. Did you specify an 'illegal' argument..?", flush=True)
        print(main.__doc__, flush=True)
        sys.exit(2)

    for opt, arg in opts: 
        if opt in ("-h", "--help"):
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--subject"):
            subject = arg
        elif opt in ("-t", "--interp"):
            interp = arg            
        elif opt in ("-i", "--img"):
            img_type = arg                 
        elif opt in ("-v", "--no_verbose"):
            verbose = False
        elif opt in ("--plot"):
            full_plot = True
        elif opt in ("--recon"):
            full_plot = True
        elif opt in ("--no_target"):
            add_target = False
        elif opt in ("--skip_img"):
            skip_img = True            
        elif opt in ("--webshow"):
            webshow = True          
        elif opt in ("--stats"):
            curve_stats = True                      
        elif opt in ("--overwrite", "-o", "--ow"):
            overwrite = True

    utils.verbose("\nslice_on_surf.py", verbose)

    # set defaults
    base_dir    = "/data1/projects/MicroFunc/Jurjen/projects/VE-pRF"
    design_dir  = opj(opd(opd(pRFline.__file__)), "data")
    fig_dir     = opj(opd(opd(pRFline.__file__)), "results")

    # fetch subject dictionary from pRFline.utils.SubjectsDict
    subj_obj = SubjectsDict()
    dict_data = subj_obj.dict_data

    if subject == "all":
        process_list = list(dict_data.keys())
    else:
        process_list = [subject]

    if full_plot:
        import imageio
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cmap_subj = "Set2"
        fig,axs = plt.subplots(ncols=len(process_list), figsize=(30,5))
        sub_colors = sns.color_palette(cmap_subj, len(process_list))

    if curve_stats:
        df_curv = []
        data_dir = opj(opd(opd(pRFline.__file__)), "data")
        fn_curv = opj(data_dir, f"sub-all_desc-curvature_{img_type}.csv")
        
    for ix,subject in enumerate(process_list):

        utils.verbose(f"\n**************************************** Processing {subject} ***************************************", verbose)
        
        # read subject-specific session from dictionary
        ses = subj_obj.get_session(subject)

        ref = opj(base_dir, "derivatives", "freesurfer", subject, "mri", "orig.nii.gz")
        if not os.path.exists(ref):
            raise FileNotFoundError(f"Could not find file '{ref}'")
        
        output_dir = os.path.dirname(ref)

        utils.verbose(f"Reference: '{ref}'", verbose)
        vol2fs_pref = f"{subject}_ses-{ses}_task-pRF"
        if img_type == "beam":
            image = opj(
                base_dir,
                subject,
                f"ses-{ses}",
                "func",
                f"{subject}_ses-{ses}_task-pRF_run-1_bold.nii.gz"
            )

            output = opj(output_dir, f"{subject}_ses-{ses}_task-pRF_space-fsnative_run-1_bold.nii.gz")
            vol2fs_suff = "run-1_bold"

        elif img_type == "recon":
            src_dir = opj(
                base_dir,
                "sourcedata",
                subject,
                f"ses-{ses}",
                "nifti"
                )

            image = utils.get_file_from_substring(["run-1", "desc-recon", ".nii.gz"], src_dir)
            interp = "lin"
            vol2fs_suff = "run-1_desc-recon"

        elif img_type == "ribbon":
            src_dir = opj(
                base_dir,
                subject,
                f"ses-{ses}",
                "func"
                )

            image = utils.get_file_from_substring(["run-1", "desc-ribbon", ".nii.gz"], src_dir)
            interp = "mul"
            vol2fs_suff = "run-1_desc-ribbon"            
        else:
            raise TypeError(f"Image type must be 'slice' or 'recon'; not '{img_type}'")

        # define volumetric output
        output = opj(output_dir, f"{vol2fs_pref}_space-fsnative_{vol2fs_suff}.nii.gz")

        # define surface output
        output_npy = opj(os.path.dirname(output), f"{subject}_ses-{ses}_task-pRF_space-fsnative_hemi-LR_{vol2fs_suff}.npy")
        if not os.path.exists(output_npy) or overwrite:
            if not isinstance(image, str):
                raise ValueError("Please specify an input file with '-i'/'--in' or '--recon'")
            utils.verbose(f"Input image: '{image}'", verbose)

            # get the registration from session to FS
            warp = utils.get_file_from_substring(
                [f"from-fs_to-ses{ses}", "genaff.mat"],
                opj(base_dir, "derivatives", "pycortex", subject, "transforms")
            )
            utils.verbose(f"Warp file: '{warp}'", verbose)

            if not isinstance(output, str):
                output = opj(os.path.basename(ref), f"{subject}_ses-{ses}_space-fsnative.nii.gz")

            # build command for antsapplytransform
            utils.verbose(f"Running 'call_antsapplytransforms'", verbose)
            cmd = f"call_antsapplytransforms -i 1 -t {interp} {ref} {image} {output} {warp}"
            os.system(cmd)

            # and call_vol2fsaverage
            utils.verbose(f"Running 'call_vol2fsaverage'", verbose)
            cmd = f"call_vol2fsaverage -p {vol2fs_pref} {subject} {output} {vol2fs_suff}"
            os.system(cmd)

            # make vertex object of output
            if not os.path.exists(output_npy):
                raise FileNotFoundError(f"Could not find file '{output_npy}'. Did 'call_vol2fsaverage' run successfully?")
        else:
            utils.verbose(f"Reading '{output_npy}'", verbose)

        data = np.load(output_npy)
        if webshow or full_plot:
            if add_target:
                data[subj_obj.get_target(subject)] = np.amax(data)*10
            data[data<1] = np.nan

            utils.verbose(f"Creating vertex object of '{output_npy}'", verbose)
            vert = pycortex.Vertex2D_fix(
                data,
                subject=subject,
                cmap="seismic",
                vmin1=0,
                vmax1=2)


            # smooth
            base = f"{subject}_ses-{ses}"
            target_npy = opj(fig_dir,subject, f"{base}_desc-target_on_surf.npy")
            if not os.path.exists(target_npy) or overwrite:
                utils.verbose(f"Creating smoothed vertex object of target vertex", verbose)
                # make smoothed version of target vertex
                target_data = np.zeros_like(data)
                target_data[subj_obj.get_target(subject)] = 1
                surf = optimal.SurfaceCalc(subject=subject)
                target_data[:surf.lh_surf_data[0].shape[0]] = surf.lh_surf.smooth(target_data[:surf.lh_surf_data[0].shape[0]],3,3)
                target_data /= target_data.max()
                np.save(target_npy, target_data)
            else:
                utils.verbose(f"Reading '{target_npy}'", verbose)
                target_data = np.load(target_npy)

            target_data[target_data<0.1] = np.nan
            target_v = pycortex.Vertex2D_fix(
                target_data,
                subject=subject,
                cmap="magma")

            # smooth

            # create data dict
            data_d = {
                f"{img_type}_on_surf": vert,
                "target_on_surf": target_v
            }

            # save image if not exists or if overwrite==True
            data_keys = list(data_d.keys())
            fn_image = opj(fig_dir, subject, f"{base}_desc-{data_keys[0]}.png")

            # initialize saving object (also opens the webviewers)
            pyc_save = pycortex.SavePycortexViews(
                data_d,
                subject=subject,
                fig_dir=opj(fig_dir,subject),
                zoom=True,
                base_name=base)

            if not skip_img:
                time.sleep(5)
                pyc_save.save_all()
        
            if full_plot:
                ax = axs[ix]

                # add image to axis
                im = imageio.imread(fn_image)
                ax.imshow(im)
                ax.set_title(
                    subject, 
                    fontsize=18, 
                    fontname="Montserrat",
                    color=sub_colors[ix], 
                    fontweight="bold")

                ax.axis('off')

        if curve_stats:

            utils.verbose(f"Extracting curvature information..", verbose)
            # get the surface information for the curvature
            surfs = optimal.SurfaceCalc(subject=subject, fs_dir=opj(base_dir, "derivatives", "freesurfer"))

            # get curvature values in mask
            curvs = surfs.surf_sm[data>0]

            df = pd.DataFrame({"curvature": curvs})
            df["subject"] = subject
            df_curv.append(df)

    if full_plot:
        fname = opj(fig_dir, f"sub-all_desc-{data_keys[0]}")
        for ext in ['png','svg']:
            utils.verbose(f"Writing {fname}.{ext}", verbose)
            fig.savefig(
                f"{fname}.{ext}",
                bbox_inches="tight",
                dpi=300,
                facecolor="white"
            )

    if curve_stats:
        if len(df_curv) > 1:
            utils.verbose(f"Writing {fn_curv}", verbose)
            df_curv = pd.concat(df_curv).to_csv(fn_curv)

if __name__ == "__main__":
    main(sys.argv[1:])

from linescanning import image, utils, transform
import os
import nibabel as nb
import numpy as np
import pickle
from bids import BIDSLayout
import matplotlib.pyplot as plt
opj = os.path.join

class Segmentations:

    def __init__(self, subject, derivatives=None, trafo_file=None, reference_slice=None, reference_session=1, target_session=2, foldover="FH"):

        self.subject            = subject
        self.derivatives        = derivatives
        self.trafo_file         = trafo_file
        self.reference_slice    = reference_slice
        self.reference_session  = reference_session
        self.target_session     = target_session
        self.foldover           = foldover
        
        # specify nighres directory
        self.nighres_dir        = opj(self.derivatives, 'derivatives', 'nighres', self.subject, f'ses-{self.reference_session}') 
        self.mask_dir           = opj(self.derivatives, 'derivatives', 'manual_masks', self.subject, f'ses-{self.reference_session}')

        # fetch segmentations, assuming default directory layout
        nighres_layout          = BIDSLayout(self.nighres_dir, validate=False).get(extension=['nii.gz'], return_type='file')
        self.wb_cruise          = utils.get_bids_file(nighres_layout, ["cortex"])
        self.wb_layers          = utils.get_bids_file(nighres_layout, ["layers"])
        self.wb_depth           = utils.get_bids_file(nighres_layout, ["depth"])

        mask_layout             = BIDSLayout(self.nighres_dir, validate=False).get(extension=['nii.gz'], return_type='file')
        self.wb_wm              = utils.get_bids_file(mask_layout, ["label-WM"])
        self.wb_gm              = utils.get_bids_file(mask_layout, ["label-GM"])
        self.wb_csf             = utils.get_bids_file(mask_layout, ["label-CSF"])
        self.wb_brainmask       = utils.get_bids_file(mask_layout, ["brainmask"])

        # check if reference slice and transformation file actually exist
        if not os.path.exists(self.reference_slice):
            raise ValueError(f"Could not find reference slice {self.reference_slice}")

        if not os.path.exists(self.trafo_file):
            raise ValueError(f"Could not find trafo_file slice {self.trafo_file}")

        # start warping (in brackets file suffixes)
        #  0 = wm prob  ("label-WM")
        #  1 = gm prob  ("label-GM")
        #  2 = csf prob ("label-CSF")
        #  3 = pve      ("cruise-cortex")
        #  4 = layers   ("layering-layers")
        #  5 = depth    ("layering-depth")
        #  6 = mask     ("brainmask")

        in_type = ['prob', 'prob', 'prob', 'tissue', 'layer', 'prob', 'tissue']
        tag = ['wm', 'gm', 'csf', 'cortex', 'layers', 'depth', 'mask']
        self.resampled = {}
        self.resampled_data ={}
        for nr,file in enumerate([self.wb_wm, self.wb_gm, self.wb_csf, self.wb_cruise, self.wb_layers, self.wb_depth, self.wb_brainmask]):

            # replace acq-MP2RAGE with acq-1slice
            new_fn = utils.replace_string(file, "acq-MP2RAGE", "acq-1slice")
            new_file = opj(self.nighres_dir, os.path.basename(new_fn))
            
            if not os.path.exists(new_file):
                if in_type[nr] == "tissue":
                    # Use MultiLabel-interpolation for tissue-segmentation
                    transform.ants_applytrafo(self.reference_slice, file, interp="mul", trafo=self.transformation_file, output=new_file)
                elif in_type[nr] == "layer":
                    # Use GenericLabel-interpolation for layer-segmentation
                    transform.ants_applytrafo(self.reference_slice, file, interp="gen", trafo=self.transformation_file, output=new_file)
                else:
                    # Use nearest neighbor-interpolation for probability maps
                    transform.ants_applytrafo(self.reference_slice, file, interp="gen", trafo=self.transformation_file, output=new_file)
            
            # collect them in 'resampled' dictionary
            self.resampled[tag[nr]] = new_file
            self.resampled_data[tag[nr]] = nb.load(new_file).get_fdata()

        self.resampled['ref'] = self.reference_slice; self.resampled_data['ref'] = nb.load(self.reference_slice).get_fdata()
        self.resampled['line'] = image.create_line_from_slice(self.reference_slice, fold=self.foldover); self.resampled_data['line'] = nb.load(self.reference_slice).get_fdata()

        pickle_file = open(opj(self.nighres_dir, f'{subject}_space-ses{self.to_ses}_desc-segmentations.pkl'), "wb")
        pickle.dump(self.resampled, pickle_file)
        pickle_file.close()

    
    def plot_segmentations(self, include=['ref', 'cortex', 'layers']):

        fig,axs = plt.subplots(figsize=(20,5))
        for ix,ax in enumerate(axs):
            segmentation_types = self.resampled[list(self.resampled.keys())[ix]]
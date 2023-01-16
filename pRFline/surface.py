from .utils import SubjectsDict
from linescanning import (
    optimal,
    prf,
    utils
)
import numpy as np
import math

class DistSurf(SubjectsDict, optimal.Neighbours):

    def __init__(
        self, 
        subject, 
        prf_epi=None,
        prf_line=None,
        hemi="lh",
        verbose=True,
        distance="geodesic",
        **kwargs):

        self.subject    = subject
        self.hemi       = hemi
        self.prf_epi    = prf_epi
        self.prf_line   = prf_line
        self.verbose    = verbose
        self.distance   = distance
        self.__dict__.update(kwargs)

        # initialize subjectsdict
        SubjectsDict.__init__(self)

        # initialize neighbours
        optimal.Neighbours.__init__(
            self,
            subject=self.subject,
            hemi=self.hemi,
            verbose=self.verbose,
            target_vert=self.get_target(self.subject)
        )

        if isinstance(self.prf_epi, str) and isinstance(self.prf_line, str):
            self.read_files()

    def read_files(self, epi=None, line=None):
            
        if not isinstance(self.prf_epi, str):
            if isinstance(epi, str):
                self.prf_epi = epi
            else:
                raise FileNotFoundError("Please specify the file with pRF-estimates from whole-brain EPI acquisition. Either through DistSurf(prf_epi='') or read_files(epi='')")
        else:
            # overwrite exiting attribute
            if self.prf_epi != epi:
                self.prf_epi = epi

        if not isinstance(self.prf_line, str):
            if isinstance(line, str):
                self.prf_line = line       
            else:
                raise FileNotFoundError("Please specify the file with pRF-estimates from line-scanning acquisition. Either through DistSurf(prf_line='') or read_files(line='')")    
        else:                     
            # overwrite exiting attribute
            if self.prf_epi != epi:
                self.prf_epi = epi

        # read in prf files
        utils.verbose(f"Reading pRF-estimate files", verbose=self.verbose)
        for ii in ["epi", "line"]:
            ff = getattr(self, f"prf_{ii}")
            if isinstance(ff, str):
                utils.verbose(f" {ii}: '{ff}'", verbose=self.verbose)
                setattr(self, f"pars_{ii}", prf.read_par_file(ff))
            else:
                raise FileNotFoundError(f"'{ff}' is not a file..")

    def find_distance(self, smooth=False, kernel=3, iterations=1):
        
        self.smooth = smooth
        self.kernel = kernel
        self.iterations = iterations
        if hasattr(self, "subsurface_verts"):
            utils.verbose(f"Target vertex was '{self.target_vert}'", verbose=self.verbose)
            # create array with all distances from V1 EPI data to line pRF
            self.dist_array = np.full(self.pars_epi.shape[0], np.nan)
            for ii in self.subsurface_verts:
                dist = prf.distance_centers(self.pars_line[0], self.pars_epi[ii,:])
                self.dist_array[ii] = dist

            # smooth data
            self.use_distance_data = self.dist_array.copy()
            if self.smooth:
                utils.verbose(f"Smoothing distance map with 'factor={self.kernel}' & 'iterations={self.iterations}'", verbose=self.verbose)
                tmp = np.full(self.pars_epi.shape[0], np.nan)
                for i in ["lh","rh"]:
                    if i == "lh":
                        sm_ = self.lh_subsurf.smooth(
                            self.dist_array[self.lh_subsurf_v], 
                            factor=self.kernel, 
                            iterations=self.iterations)

                        tmp[self.lh_subsurf_v] = sm_
                    elif i == "rh":
                        sm_ = self.rh_subsurf.smooth(
                            self.dist_array[self.rh_subsurf_v], 
                            factor=self.kernel, 
                            iterations=self.iterations)
                        
                        tmp[self.rh_subsurf_v] = sm_

                self.dist_array_sm = tmp.copy()
                self.use_distance_data = self.dist_array_sm.copy()

            # get the vertex closest to the line pRF
            self.closest_to_minimum_distance = np.where(self.use_distance_data == np.nanmin(self.use_distance_data))[0][0]
            self.dva_dist = self.use_distance_data[self.closest_to_minimum_distance]
            utils.verbose(f"Smallest difference = {round(self.dva_dist,2)}dva (vertex='{self.closest_to_minimum_distance}')", verbose=self.verbose)

            # get the distance between that vertex & target vertex
            active_hemi = getattr(self, f"{self.hemi}_dist_to_targ")
            self.dist_geodesic = active_hemi[self.closest_to_minimum_distance]

            utils.verbose(f"Geodesic distance '{self.closest_to_minimum_distance}' to '{self.target_vert}' = {round(self.dist_geodesic,2)}mm", verbose=self.verbose)

            # get euclidian distance
            surf = getattr(self, f"{self.hemi}_surf_data")[0]
            self.coord_targ = surf[self.target_vert]
            self.coord_match = surf[self.closest_to_minimum_distance]
            self.dist_euclidian = math.dist(self.coord_targ,self.coord_match)

            utils.verbose(f"Euclidian distance '{self.closest_to_minimum_distance}' to '{self.target_vert}' = {round(self.dist_euclidian,2)}mm", verbose=self.verbose)            

    def make_vertex(self, **kwargs):
        
        import cortex
        
        # set defaults
        for xx,qq in zip(["annotate","annotate_value","vmin","vmax","cmap"], [True,30,0,4,"magma_r"]):
            if not hasattr(self, xx):
                setattr(self, xx, qq)

            if xx in ["vmax","vmin","cmap"]:
                if xx in list(kwargs.keys()):
                    utils.verbose(f"Updating '{xx}' from '{getattr(self,xx)}' to '{kwargs[xx]}'", verbose=self.verbose)
                    setattr(self, xx, kwargs[xx])
                    kwargs.pop(xx)
                else:
                    setattr(self, xx, qq)

        # fill target vertex & best-matching vertex
        if self.annotate:
            self.use_distance_data[self.closest_to_minimum_distance] = self.annotate_value
            self.use_distance_data[self.target_vert] = self.annotate_value

            self.target_matched = np.full_like(self.use_distance_data, 0)
            self.target_matched[self.closest_to_minimum_distance] = -self.annotate_value
            self.target_matched[self.target_vert] = self.annotate_value

            # make vertex objects
            self.target_matched_v = cortex.Vertex(
                self.target_matched, 
                **dict(
                    kwargs,
                    subject=self.subject, 
                    vmin=-10, 
                    vmax=10,
                    cmap="seismic",
                ))

        # make vertex objects
        self.dist_v = cortex.Vertex(
            self.use_distance_data, 
            **dict(
                kwargs,
                subject=self.subject, 
                vmin=self.vmin, 
                vmax=self.vmax,
                cmap=self.cmap,
            ))

        if self.smooth:
            self.dist_v_raw = cortex.Vertex(
                self.dist_array, 
                **dict(
                    kwargs,
                    subject=self.subject, 
                    vmin=self.vmin, 
                    vmax=self.vmax,
                    cmap=self.cmap,
                ))            

    def webshow(self, **kwargs):
        import cortex
        if not hasattr(self, "dist_v"):
            self.make_vertex(**kwargs)

        data = {"distance": self.dist_v}
        if hasattr(self, "target_matched_v"):
            data["targ-match"] = self.target_matched_v

        if hasattr(self, "dist_v_raw"):
            data["no smoothing"] = self.dist_v_raw

        cortex.webgl.show(data)


class PredpRF(prf.pRFmodelFitting):

    def __init__(
        self,
        prf_params,
        model="norm",
        TR=1,
        verbose=False,
        grid_size=20,
        vf_extent=[-5,5],
        dm=None,
        **kwargs):
    
        self.prf_params = prf_params
        self.grid_size  = grid_size
        self.model      = model
        self.TR         = TR
        self.verbose    = verbose
        self.vf_extent  = vf_extent
        self.dm         = dm
        self.__dict__.update(kwargs)

        if isinstance(self.dm, (str,np.ndarray)):
            self.dm = prf.read_par_file(self.dm)
        else:
            self.create_matrix()

        prf.pRFmodelFitting.__init__(
            self,
            None,
            design_matrix=self.dm,
            TR=self.TR,
            verbose=self.verbose,
            model=self.model,
            hrf="direct"
        )

        self.load_params(
            self.prf_params,
            model=self.model
        )

        self.get_prediction(**kwargs)
        if hasattr(self, "coords"):
            self.get_distance_to_prf()

    def get_prediction(self, **kwargs):
        _,self.rf,_,self.pred = self.plot_vox(
            model=self.model, 
            # resize_pix=self.grid_size, 
            **kwargs)

    def get_distance_to_prf(self):
        self.dist_to_prf = np.zeros(len(self.coords))
        for ix,ii in enumerate(self.coords):
            self.dist_to_prf[ix] = prf.distance_centers(self.prf_params, ii)

    def create_matrix(self):

        self.dm = np.zeros((self.grid_size,self.grid_size,(self.grid_size*self.grid_size)))
        start = 0
        self.coords = []
        self.coords_ix = []
        self.span = sum(abs(n) for n in self.vf_extent)
        for x in range(self.dm.shape[1]):
            for y in range(self.dm.shape[0]):
                self.dm[x,y,start] = 1
                start += 1

                # get the coordinates of pixels
                center = (
                    self.vf_extent[0]+(y*self.span/self.grid_size)+((self.span/self.grid_size)/2),
                    self.vf_extent[1]-(x*self.span/self.grid_size)-((self.span/self.grid_size)/2)
                )
                self.coords.append(center)
                self.coords_ix.append([x,y])
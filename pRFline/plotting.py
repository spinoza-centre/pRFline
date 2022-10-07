import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from PIL import ImageColor
from linescanning import (
    prf,
    utils,
    plotting
)
import pickle
opj = os.path.join

class pRFSpread():

    def __init__(
        self,
        fits, 
        subject=None,
        model="gauss", 
        fig_dir=None, 
        data_type="lines",
        colors=["#DE3163", "#6495ED"],
        fct=6,
        lim_dva=None,
        lim_size=None,
        lim_r2=None,
        lim_fits=None,
        max_fontsize=24,
        **kwargs):

        self.fits       = fits
        self.sub        = subject
        self.model      = model
        self.fig_dir    = fig_dir
        self.data_type  = data_type
        self.colors     = colors
        self.fct        = fct
        self.lim_dva    = lim_dva
        self.lim_size   = lim_size
        self.lim_r2     = lim_r2
        self.lim_fits   = lim_fits
        self.max_size   = max_fontsize
        self.__dict__.update(**kwargs)

        # for attr,elem in zip(["dva", "size", "r2"], [self.lim_dva, self.lim_size, self.lim_r2]):
        #     if not hasattr(self, f"ticks_{attr}"):
        #         ticks = list(np.arange(*elem))

    def plot_fits(self, **kwargs):

        self.fct = 6
        self.fit_fig = plt.figure(
            constrained_layout=True, 
            figsize=(self.fct*4,len(self.fits)*self.fct))

        gs = self.fit_fig.add_gridspec(
            len(self.fits), 
            2, 
            wspace=0.01,
            width_ratios=[35,100])

        for ix,obj in enumerate(self.fits):
            if obj == "avg":
                lbl = f"average; {[round(ii,2) for ii in self.fits[obj].gauss_iter[0]]}"
            else:
                lbl = f"run-{ix+1}; {[round(ii,2) for ii in self.fits[obj].gauss_iter[0]]}"

            ax1 = self.fit_fig.add_subplot(gs[ix,0])
            ax2 = self.fit_fig.add_subplot(gs[ix,1])
            _,_,_,_ = self.fits[obj].plot_vox(
                axs=[ax1,ax2],
                model=self.model,
                stage="iter",
                axis_type="time",
                title=lbl,
                y_lim=self.lim_fits,
                font_size=self.max_size*0.8,
                label_size=self.max_size*0.5,
                resize_pix=270,
                **kwargs
            )

            self.fit_fig.suptitle(
                f"{self.sub}: fits of individual runs ({len(self.fits)-1})",
                fontsize=self.max_size,
                x=0.6)

    def plot_spread(self, **kwargs):

        # create color palette
        self.cmap1 = utils.make_binary_cm(self.colors[0])
        self.color_p = sns.dark_palette(self.colors[1], len(self.fits)-1, reverse=True)
        self.color_p += [tuple([ii/255 for ii in ImageColor.getcolor(self.colors[0], "RGB")])]
        
        self.spread_fig = plt.figure(figsize=(24,6))
        self.gs = self.spread_fig.add_gridspec(1,5, width_ratios=(35,25,10,15,15), wspace=0.35)
        self.fig_axs = [self.spread_fig.add_subplot(self.gs[ii]) for ii in range(5)]

        self.dist = []
        # self.data = {}
        for ii in self.fits:

            if ii != "avg":

                # get parameters from individual runs
                pars,_,_,_ = self.fits[ii].plot_vox(
                    model=self.model,
                    stage='iter',
                    make_figure=False
                )
            
                # get parameters from average
                avg_pars,_,_,_ = self.fits['avg'].plot_vox(
                    model=self.model,
                    stage='iter',
                    make_figure=False
                )

                # make prf object of average
                avg_prf = prf.make_prf(
                    self.fits['avg'].prf_stim, 
                    mu_x=0,
                    mu_y=0,
                    size=1,
                    resize_pix=500
                )

                # plot
                if ii == 0:
                    prf1 = np.zeros_like(avg_prf)

                    # create circle object
                    circ = plt.Circle(
                        (0,0),
                        1,
                        fill=False,
                        ec=self.colors[0],
                        lw=3)

                    self.fig_axs[0].set_aspect(1)
                    self.fig_axs[0].add_artist(circ)

                    # create visual field delineation
                    plotting.LazyPRF(
                        prf1, 
                        [-5,5],
                        ax=self.fig_axs[0],
                        cross_color="k",
                        shrink_factor=0.9,
                        cmap=self.cmap1,
                        title="2D pRF location\n(normalized to average)",
                        vf_only=True,
                        font_size=self.max_size*0.8,
                        alpha=0.8,
                        **kwargs)

                    # get 1d profile
                    avg_1d = prf.gauss_1d_function(
                        self.fits[ii].prf_stim.x_coordinates[0],
                        avg_pars[0],
                        avg_pars[2]
                    )

                # plot normalized pRF
                run_pars = prf.normalize_prf(avg_pars,pars)
                run_prf = prf.make_prf(
                    self.fits[ii].prf_stim, 
                    mu_x=run_pars[0],
                    mu_y=run_pars[1],
                    size=run_pars[2],
                    resize_pix=500
                )
                
                # find x/y of max pRF
                x,y = np.where(run_prf == np.amax(run_prf))
                center = np.array([x[0],y[0]])

                # convert pixels to whatever visual field is
                center = tuple(center/run_prf.shape[0]*10-5)

                circ2 = plt.Circle(
                    center,
                    run_pars[2],
                    fill=False,
                    ec=self.color_p[ii],
                    lw=2)

                # ax1.set_aspect(1)
                self.fig_axs[0].add_artist(circ2)

                # get 1d profile
                run_1d = prf.gauss_1d_function(
                    self.fits[ii].prf_stim.x_coordinates[0],
                    pars[0],
                    pars[2]
                )

                plotting.LazyPlot(
                    run_1d,
                    axs=self.fig_axs[1],
                    line_width=2,
                    color=self.color_p[ii]
                )

                # get euclidian distance between average pRF and run-pRF
                self.dist.append(prf.distance_centers(avg_pars, pars))

        plotting.LazyPlot(
            avg_1d,
            axs=self.fig_axs[1],
            line_width=2,
            color=self.colors[0],
            title="1D pRF location",
            font_size=self.max_size*0.8,
            y_ticks=[0,1],
            x_label="visual field (px)",
            y_label="response",
            x_lim=[0,avg_1d.shape[0]],
            x_ticks=[0,avg_1d.shape[0]//2,avg_1d.shape[0]],
            **kwargs
        )

        self.fig_axs[1].legend([f"run-{ii+1}" if isinstance(ii,int) else ii for ii in list(self.fits.keys())], frameon=False)

        # make plot with spread in dva
        self.dist = np.array(self.dist)

        if not isinstance(self.lim_dva, list):
            self.lim_dva = [0,round(np.amax(self.dist),2)]

        plotting.LazyBar(
            x=[ii+1 for ii in list(self.fits.keys()) if ii != "avg"],
            y=self.dist,
            palette=self.color_p,
            axs=self.fig_axs[2],
            sns_ori="v",
            lim=self.lim_dva,
            add_labels=True,
            x_label2="run ID",
            y_label2="dva",
            font_size=self.max_size*0.8,
            title2="Distance from average",
            ticks=[0,self.lim_dva[-1]],
        )

        # make plot with size
        self.sizes = np.array([getattr(self.fits[ii], f"{self.model}_iter")[:,2][0] for ii in self.fits])
        size_colors = self.color_p+[tuple([ii/255 for ii in ImageColor.getcolor(self.colors[0], "RGB")])]

        if not isinstance(self.lim_size, list):
            self.lim_size = [0,round(np.amax(self.sizes),2)]

        plotting.LazyBar(
            x=[ii+1 if isinstance(ii,int) else ii for ii in list(self.fits.keys())],
            y=self.sizes,
            palette=size_colors,
            axs=self.fig_axs[3],
            sns_ori="v",
            lim=self.lim_size,
            add_labels=True,
            x_label2="run ID",
            y_label2="dva",
            font_size=self.max_size*0.8,
            title2="pRF size",
            ticks=[0,self.lim_size[-1]],
            **kwargs
        )

        # make plot with r2
        self.r2 = np.array([getattr(self.fits[ii], f"{self.model}_iter")[:,-1][0] for ii in self.fits])
        if not isinstance(self.lim_r2, list):
            self.lim_r2 = [0,round(np.amax(self.r2),2)]

        plotting.LazyBar(
            x=[f"{ii+1}" if isinstance(ii,int) else ii for ii in list(self.fits.keys())],
            y=self.r2,
            palette=size_colors,
            axs=self.fig_axs[4],
            sns_ori="v",
            lim=self.lim_r2,
            add_labels=True,
            x_label2="run ID",
            y_label2="r2",
            font_size=self.max_size*0.8,
            title2="Variance explained",
            ticks=[0,self.lim_r2[-1]],
            **kwargs
        )    

        if self.data_type == "lines":
            self.title_txt = "line-scanning"
            self.fig_txt = "lines"
        else:
            self.title_txt = "2D EPI"
            self.fig_txt = "2depi"

        self.spread_fig.suptitle(f"Spread in {self.title_txt} data", fontsize=self.max_size, y=1.05)

    def save_fig(self):
        if hasattr(self, "spread_fig"):
            fname = opj(self.fig_dir, f"{self.sub}_model-{self.model}_desc-spread_{self.fig_txt}.svg")
            print(f"Writing {fname}")
            self.spread_fig.savefig(
                fname, 
                dpi=300, 
                bbox_inches='tight', 
                facecolor="white")

        if hasattr(self, "fit_fig"):
            fname = opj(self.fig_dir, f"{self.sub}_model-{self.model}_desc-fits_{self.fig_txt}.svg")
            print(f"writing {fname}")
            self.fit_fig.savefig(
                fname, 
                dpi=300, 
                bbox_inches='tight', 
                facecolor="white")            

    def save_pars(self):
        # save parameter objects
        self.par_file = opj(self.fig_dir, f"{self.sub}_model-{self.model}_desc-spread_{self.fig_txt}.pkl")
        if not os.path.exists(self.par_file):
            with open(self.par_file, 'wb') as handle:
                pickle.dump(self.fits, handle)
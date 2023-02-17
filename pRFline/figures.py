from linescanning import(
    plotting,
    utils,
    prf,
    glm,
    fitting
)
from itertools import repeat
from sklearn.metrics import r2_score
import pRFline
from pRFline.utils import (
    SubjectsDict,
    read_subject_data)
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
from matplotlib import (
    patches, 
    lines)
import pandas as pd
import numpy as np
import imageio
import seaborn as sns
import string

opj = os.path.join
opd = os.path.dirname

class MainFigure():

    def __init__(
        self,
        full_dict=None,
        figsize=(24,10),
        deriv=None,
        model="gauss",
        verbose=False,
        cmap="Set2",
        targ_match_colors=["r","b"],
        ):
        
        self.figsize = figsize
        self.full_dict = full_dict
        self.deriv = deriv
        self.model = model
        self.verbose = verbose
        self.cmap = cmap
        self.targ_match_colors = targ_match_colors
        self.results_dir = opj(opd(opd(pRFline.__file__)), "results")
        self.data_dir = opj(opd(opd(pRFline.__file__)), "data")

        # don't inherit to keep it more separate
        self.plot_defaults = plotting.Defaults()
        self.subj_obj = SubjectsDict()
        self.process_subjs = self.subj_obj.get_subjects()

        # set color palette & create empty color map for overlap plots
        self.sub_colors = sns.color_palette(self.cmap, len(self.process_subjs))
        self.empty_cmap = utils.make_binary_cm("#cccccc")

        # set derivatives
        if not isinstance(deriv, str):
            self.deriv = os.environ.get("DIR_DATA_DERIV")

        # fetch data if it's not a dictionary
        fetch_data = True
        if isinstance(self.full_dict, str):
            if os.path.exists(self.full_dict):
                utils.verbose(f"Reading '{self.full_dict}'", self.verbose)
                if self.full_dict.endswith("csv"):
                    fetch_data = False
                    self.df_params = pd.read_csv(self.full_dict).set_index(["subject","acq","code","run"])
                else:
                    raise TypeError(f"Not sure what to do with this file.. Please specify a 'csv'-file, not {self.full_dict}")
        elif isinstance(self.full_dict, pd.DataFrame):
            fetch_data = False
            self.df_params = self.full_dict.copy()

        # check if the flag was switched to True
        if fetch_data:
            self.fetch_data()
            self.fetch_parameters()

        # get uppercase letters for annotations
        self.alphabet = list(string.ascii_uppercase)

    def fetch_data(self):
        
        utils.verbose("Reading data. Can take a while..", self.verbose)
        self.full_dict = {}
        n_jobs = len(self.process_subjs)
        dd = Parallel(n_jobs=n_jobs,verbose=False)(
            delayed(read_subject_data)(
                subject,
                deriv=self.deriv,
                model=self.model,
                fix_bold=True,
                verbose=False
            )
            for subject in self.process_subjs
        )

        for ix,subject in enumerate(self.process_subjs):
            self.full_dict[subject] = dd[ix]
        
    def fetch_parameters(self):

        utils.verbose("Creating full dataframe of pRF estimates..", self.verbose)

        self.df_params = []
        for sub in self.process_subjs:

            avg_obj = self.get_wb_avg(subject=sub)
            run_obj = self.get_wb_runs(subject=sub)
            line_obj = self.get_line_avg(subject=sub)
            rib_obj = self.get_line_ribbon(subject=sub)
            full_obj = self.get_line_all(subject=sub)

            # parse into dataframe
            wb_avg = getattr(avg_obj, f"{self.model}_iter")[self.subj_obj.get_target(sub),:]
            wb_df = prf.Parameters(wb_avg, model=self.model).to_df()
            wb_df["subject"],wb_df["run"],wb_df["acq"], wb_df["code"] = sub, "avg", "wb avg", 0

            self.df_params.append(wb_df)

            # add HRF parameters if they don't exist; ensure same shapes across wb/lines
            for ii in ["hrf_deriv", "hrf_disp"]:
                if ii not in list(wb_df.keys()):
                    wb_df[ii] = np.nan

            for run in range(run_obj.data.shape[0]):
                pars,_,_,_ = run_obj.plot_vox(
                    vox_nr=run,
                    model=run_obj.model,
                    stage='iter',
                    make_figure=False
                )

                pars_df = prf.Parameters(pars, model=self.model).to_df()
                pars_df["subject"],pars_df["run"],pars_df["acq"],pars_df["code"] = sub, run+1,"wb runs",1

                self.df_params.append(pars_df)

            # do lines
            avg_line,_,_,_ = line_obj.plot_vox(
                model=line_obj.model,
                stage='iter',
                make_figure=False
            )

            line_df = prf.Parameters(avg_line, model=self.model).to_df()
            line_df["subject"],line_df["run"],line_df["acq"],line_df["code"] = sub, "avg", "line avg", 2

            self.df_params.append(line_df)

            # do ribbon
            for depth in range(rib_obj.data.shape[0]):
                pars,_,_,_ = rib_obj.plot_vox(
                    vox_nr=depth,
                    model=run_obj.model,
                    stage='iter',
                    make_figure=False
                )

                pars_df = prf.Parameters(pars, model=self.model).to_df()
                pars_df["subject"],pars_df["run"],pars_df["acq"],pars_df["code"] = sub, f"depth-{depth+1}","ribbon",3

                self.df_params.append(pars_df)

            # do full line
            pars = getattr(full_obj, f"{self.model}_iter").copy()
            vox_range = np.arange(pars.shape[0])
            pars_df = prf.Parameters(pars, model=self.model).to_df()
            pars_df["subject"],pars_df["run"],pars_df["acq"],pars_df["code"] = sub, [f"vox-{ii}" for ii in vox_range],"full",4

            self.df_params.append(pars_df)

        self.df_params = pd.concat(self.df_params).set_index(["subject","acq","code","run"])

        df_fname = opj(self.data_dir, f"sub-all_model-{self.model}_desc-full_params.csv")
        if not os.path.exists(df_fname):
            utils.verbose(f"Writing '{df_fname}'", self.verbose)
            self.df_params.to_csv(df_fname)

    def plot_r2(
        self, 
        axs=None, 
        include_ribbon=False,
        posthoc=False):

        if axs == None:
            _,axs = plt.subplots(figsize=(4,8))

        if not hasattr(self, "df_params"):
            self.fetch_parameters()

        if not include_ribbon:
            data = utils.select_from_df(self.df_params, expression=("acq != ribbon", "&", "acq != full"))
        else:
            data = self.df_params.copy()

        self.df_r2 = data.reset_index().sort_values(['code','subject'])
        self.r2_plot = plotting.LazyBar(
            data=self.df_r2,
            x="acq",
            y="r2",
            sns_ori="v",
            sns_rot=-20,
            sns_offset=5,
            axs=axs,
            add_labels=True,
            color="#cccccc",
            add_points=True,
            points_cmap=self.sub_colors,
            points_hue="subject",    
            y_label2="variance explained (r2)",
            lim=[0,1],
            ticks=list(np.arange(0,1.2,0.2)),
            fancy=True,
            trim_bottom=True)

        # run posthoc?
        if posthoc:
            self.posth = glm.Posthoc(
                df=self.df_r2,
                dv="r2",
                between="acq",
                axs=axs,
                annotate_ns=True)

            self.posth.plot_bars()

    def compile_figure(
        self,
        img_dist=None,
        coord_targ=(1594,3172),
        coord_closest=(1594,3205),
        csv_file=None,
        include=["euclidian","geodesic"],
        fontsize=28,
        save_as=None,
        inset_axis=[0.6,-0.25,0.7,0.7],
        inset_extent=[1000,2600,2400,3400],
        cbar_inset=[0.1,1.1,0.8,0.05],
        txt_pos1=(-200,35),
        txt_pos2=(-50,-80),
        flip_ticks=False,
        flip_label=False,
        clip_x=3900):

        # initiate full figure
        self.fig = plt.figure(figsize=(24,10))
        self.subfigs = self.fig.subfigures(nrows=2, height_ratios=[0.5,1])
        self.row1 = self.subfigs[0].subplots(ncols=len(self.process_subjs))
        self.row2 = self.subfigs[1].subplots(
            ncols=3+len(include), 
            gridspec_kw={
                'width_ratios': [0.6,0.2,0.6]+[0.05 for ii in range(len(include))], 
                'wspace': 0.6})

        # plot subject-specific overlap
        self.plot_overlap(axs=self.row1)

        # plot normalized pRFs
        self.plot_norm_overlap(axs=self.row2[0])

        # plot r2
        self.plot_r2(axs=self.row2[1])        

        # plot surface
        self.img_dist = img_dist

        self.plot_surfaces(
            img_dist=self.img_dist,
            coord_targ=coord_targ,
            coord_closest=coord_closest,
            axs=self.row2[2],
            inset_axis=inset_axis,
            inset_extent=inset_extent,
            cbar_inset=cbar_inset,
            txt_pos1=txt_pos1,
            txt_pos2=txt_pos2,
            flip_ticks=flip_ticks,
            flip_label=flip_label,
            clip_x=clip_x)

        # plot distances
        self.plot_surface_distances(
            axs=self.row2[-len(include):],
            csv_file=csv_file,
            include=include
        )

        # tight layout befor annotations
        plt.tight_layout()

        # make annotations
        dist_from_x0 = 1/(self.fig.get_size_inches()*self.fig.dpi)[0]*50
        self.row1[0].annotate(
            "A", 
            (dist_from_x0,1.05), 
            fontsize=fontsize, 
            xycoords="figure fraction")

        for ix,ax in enumerate(self.row2):

            if ix == 0:
                pos = dist_from_x0
            else:
                pos = self.row2[ix].get_position().x0-dist_from_x0

            ax.annotate(
                self.alphabet[ix+1], 
                (pos,0.65), 
                fontsize=fontsize, 
                xycoords="figure fraction")

        # save figure?
        if isinstance(save_as, str):
            for ext in ["png","svg"]:

                fname = f"{save_as}.{ext}"
                utils.verbose(f"Writing '{fname}'", self.verbose)

                self.fig.savefig(
                    fname,
                    bbox_inches="tight",
                    dpi=300,
                    facecolor="white"
                )

        plt.show()

    def plot_surfaces(
        self,
        img_dist=None,
        axs=None,
        coord_targ=(1594,3172),
        coord_closest=(1594,3205),
        inset_axis=[0.6,-0.25,0.7,0.7],
        inset_extent=[1000,2600,2400,3400],
        cbar_inset=[0.1,1.1,0.8,0.05],
        txt_pos1=(-50,15),
        txt_pos2=(0,-25),
        vmin=0,
        vmax=4,
        dist_cm="magma_r",
        flip_ticks=False,
        flip_label=False,
        clip_x=3900):

        if axs == None:
            _,axs = plt.subplots(figsize=(8,8))

        img_d = imageio.v2.imread(img_dist)
        axs.imshow(img_d)
        x1, x2, y1, y2 = inset_extent
        axs.set_xlim(x1,x2)
        axs.set_ylim(y2,y1)
        axs.set_xticklabels([])
        axs.set_yticklabels([])    
        axs.axis('off')

        # annotate
        for cc,col,hh,pos in zip(
            [coord_targ,coord_closest], # coordinates
            self.targ_match_colors,     # colors
            ["target","match"],         # text
            [txt_pos1,txt_pos2]):       # text position

            # add dot at vertex location
            circ = plt.Circle(
                cc,
                15,
                fc=col,
                fill=True)

            axs.add_artist(circ)

            # add labels
            axs.annotate(
                hh,
                color=col,
                fontweight="bold",
                fontsize=self.plot_defaults.font_size,
                xy=cc, 
                xycoords='data',
                xytext=pos, 
                textcoords='offset pixels')            

        axs3 = axs.inset_axes(inset_axis)
        if isinstance(clip_x, int):
            axs3.imshow(img_d[:,:clip_x,:])
        else:
            axs3.imshow(img_d)
        
        # annotate
        for cc,col in zip(
            [coord_targ,coord_closest], # coordinates
            self.targ_match_colors):    # colors

            # add dot at vertex location
            circ = plt.Circle(
                cc,
                15,
                fc=col,
                fill=True)    

            axs3.add_artist(circ)

        rect, lines = axs3.indicate_inset_zoom(axs, ec="#cccccc")

        coords = rect.get_patch_transform().transform(rect.get_path().vertices[:-1])
        lines[0].xy2 = coords[0]
        lines[3].xy2 = coords[2]

        axs3.axis('off')

        if isinstance(cbar_inset, list):
            cbar_ax = axs.inset_axes(cbar_inset)
            plotting.LazyColorbar(
                cbar_ax,
                cmap=dist_cm,
                txt="distance to target [dva]",
                vmin=vmin,
                vmax=vmax,
                ori="horizontal",
                flip_ticks=flip_ticks,
                flip_label=flip_label
            )

    def plot_surface_distances(
        self,
        csv_file=None,
        include=["euclidian","geodesic","dva"],
        axs=None,
        subject=None):

        if isinstance(csv_file, str):
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"Could not find csv-file with distance measures on the surface. See 'pRFline/scripts/dist_on_surf.py'")
        else:
            raise ValueError("Please specify a csv-file with distance measures on the surface. See 'pRFline/scripts/dist_on_surf.py'")

        if isinstance(include, str):
            include = [include]
        
        if not isinstance(axs, (list,np.ndarray)):
            _,axs = plt.subplots(
                ncols=len(include), 
                figsize=(len(include)*2,8),
                gridspec_kw={
                    "wspace": 0.9
                })

        # select particular subject or remove sub-009
        self.df_surf = pd.read_csv(csv_file, index_col=0)
        if isinstance(subject, str):
            self.df_surf = utils.select_from_df(self.df_surf, expression=f"subject = {subject}")
        else:
            self.df_surf = utils.select_from_df(self.df_surf, expression="subject != sub-009")

        for ix,par in enumerate(include):

            if par == "dva":
                y_lim = [0,1]
                y_lbl = f"distance target-match [dva]"
            else:
                y_lim = [0,12]
                y_lbl = f"distance target-match [{par}; mm]"
            
            ax = axs[ix]
            plotting.LazyBar(
                data=self.df_surf,
                x="ix",
                y=par,
                sns_ori="v",
                axs=ax,
                sns_offset=4,
                color="#cccccc",
                add_points=True,
                points_cmap=self.cmap,
                points_hue="subject",    
                y_label2=y_lbl,
                fancy=True,
                lim=y_lim,
                trim_bottom=True
            )

    def plot_norm_overlap(
        self,
        axs=None,
        vf_extent=[-5,5]):

        if axs == None:
            _,axs = plt.subplots(figsize=(8,8))

        if not hasattr(self, "df_params"):
            self.fetch_parameters()
        
        plotting.LazyPRF(
            np.zeros((500,500)), 
            vf_extent,
            ax=axs,
            cross_color="k",
            edge_color=None,
            shrink_factor=0.9,
            cmap=self.empty_cmap,
            vf_only=True)

        # do all runs first
        for ix,sub in enumerate(self.process_subjs):

            # get subject-specific dataframe
            sub_pars = utils.select_from_df(self.df_params, expression=f"subject = {sub}")
            run_pars = utils.select_from_df(sub_pars, expression=f"code = 1")

            # get wb average
            avg_wb = utils.select_from_df(sub_pars, expression="code = 0").iloc[0,:].values

            # loop through runs
            for run in range(run_pars.shape[0]):

                pars = run_pars.iloc[run,:].values
                run_norm = prf.normalize_prf(avg_wb,pars)

                center = (run_norm[0],run_norm[1])
                circ = plt.Circle(
                    center,
                    run_norm[2],
                    ec="#cccccc",
                    fill=False,
                    alpha=0.7)

                axs.add_artist(circ)

        # create black circle with SD=1
        circ_avg = plt.Circle(
            (0,0),
            1,
            ec="k",
            fill=False,
            lw=2)

        axs.add_artist(circ_avg)

        # add normalized subject-specific ones
        for ix,sub in enumerate(self.process_subjs):

            # get subject-specific dataframe
            sub_pars = utils.select_from_df(self.df_params, expression=f"subject = {sub}")
            
            # extract average wb/line-scanning pRFs
            avg_wb = utils.select_from_df(sub_pars, expression="code = 0").iloc[0,:].values
            avg_line = utils.select_from_df(sub_pars, expression="code = 2").iloc[0,:].values

            # normalize
            line_norm = prf.normalize_prf(avg_wb,avg_line)
            center = (line_norm[0],line_norm[1])
            circ3 = plt.Circle(
                center,
                line_norm[2],
                ec=self.sub_colors[ix],
                fill=False,
                lw=3)

            axs.add_artist(circ3)

        # add axis annotations
        for ii,val in zip(["-5sd","5sd","-5sd","5sd"], [(0,0.51),(0.98,0.51),(0.51,0),(0.51,0.98)]):
            axs.annotate(
                ii,
                val,
                fontsize=self.plot_defaults.label_size,
                xycoords="axes fraction"
            )            

    def plot_overlap(
        self, 
        axs=None,
        vf_extent=[-5,5],
        fontsize=18):

        if not isinstance(axs, (list,np.ndarray)):
            _,axs = plt.subplots(ncols=len(self.process_subjs), figsize=(24,5))
        else:
            if len(axs) != len(self.process_subjs):
                raise ValueError(f"Number of axes ({len(axs)}) must match number of subjects ({len(self.process_subjs)})")

        if not hasattr(self, "df_params"):
            self.fetch_parameters()

        for ix,sub in enumerate(self.process_subjs):
            
            # initiate RF figure
            plotting.LazyPRF(
                np.zeros((500,500)), 
                vf_extent,
                ax=axs[ix],
                cmap=self.empty_cmap,
                cross_color="k",
                edge_color=None,
                shrink_factor=0.9,
                vf_only=True)

            sub_pars = utils.select_from_df(self.df_params, expression=f"subject = {sub}")
            run_pars = utils.select_from_df(sub_pars, expression=f"code = 1")
            # loop through runs
            for run in range(run_pars.shape[0]):

                pars = run_pars.iloc[run,:].values
                if pars[-1] != 0:         
                    # add non-normalized to subject specific axis
                    circ_run_subj = plt.Circle(
                        (pars[0],pars[1]),
                        pars[2],
                        ec="#cccccc",
                        fill=False,
                        alpha=0.7)

                    axs[ix].add_artist(circ_run_subj)

            # wb pRF
            avg_wb = utils.select_from_df(sub_pars, expression="code = 0").iloc[0,:].values
            sub_wb = plt.Circle(
                (avg_wb[0],avg_wb[1]),
                avg_wb[2],
                ec="k",
                fill=False,
                lw=2)

            axs[ix].add_artist(sub_wb)    

            # line pRF
            avg_line = utils.select_from_df(sub_pars, expression="code = 2").iloc[0,:].values
            sub_line = plt.Circle(
                (avg_line[0],avg_line[1]),
                avg_line[2],
                ec=self.sub_colors[ix],
                fill=False,
                lw=2)

            axs[ix].add_artist(sub_line)
            axs[ix].set_title(
                sub, 
                fontsize=fontsize, 
                color=self.sub_colors[ix], 
                fontweight="bold")

        for ii,val in zip(["-5°","5°","-5°","5°"], [(0,0.51),(0.98,0.51),(0.51,0),(0.51,0.96)]):
            axs[0].annotate(
                ii,
                val,
                fontsize=self.plot_defaults.label_size,
                xycoords="axes fraction"
            )


    def get_wb_runs(self, subject=None):
        return self.full_dict[subject]['wb']['runs']

    def get_wb_avg(self, subject=None):
        return self.full_dict[subject]['wb']['avg']

    def get_line_avg(self, subject=None):
        return self.full_dict[subject]['lines']['avg']

    def get_line_ribbon(self, subject=None):
        return self.full_dict[subject]['lines']['ribbon']

    def get_line_all(self, subject=None):
        return self.full_dict[subject]['lines']['all']        

class CrossBankFigure(MainFigure):

    def __init__(
        self,
        **kwargs):

        self.__dict__.update(kwargs)
        super().__init__(**kwargs)
    
    def plot_bar_comparison(
        self,
        axs=None,
        csv_file=None,
        include=["euclidian","geodesic"],
        cmap="inferno",
        move_box=False,
        leg_anchor=(1.8,1)):

        if isinstance(csv_file, str):
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"Could not find csv-file with distance measures on the surface. See 'pRFline/scripts/dist_on_surf.py'")
        else:
            raise ValueError("Please specify a csv-file with distance measures on the surface. See 'pRFline/scripts/dist_on_surf.py'")

        if isinstance(include, str):
            include = [include]
        
        if not isinstance(axs, (list,np.ndarray)):
            _,axs = plt.subplots(
                ncols=len(include), 
                figsize=(len(include)*2,8),
                gridspec_kw={
                    "wspace": 0.9
                })

        self.df_surf = pd.read_csv(csv_file, index_col=0)
        for ix,par in enumerate(include):

            if par == "dva":
                y_lim = [0,1]
                y_lbl = f"dva"
            else:
                y_lim = [0,10]
                y_lbl = f"{par} [mm]"
            
            if ix == 0:
                leg = True
                anchor = leg_anchor
            else:
                leg = False
                anchor = None

            ax = axs[ix]
            plotting.LazyBar(
                data=self.df_surf,
                x="ix",
                y=par,
                sns_ori="v",
                axs=ax,
                sns_offset=4,
                color="#cccccc",
                add_points=True,
                points_cmap=cmap,
                points_hue="smooth_lbl",
                points_legend=leg,
                y_label2=y_lbl,
                fancy=True,
                lim=y_lim,
                trim_bottom=True,
                bbox_to_anchor=anchor,
            )

        if move_box:
            box = axs[0].get_position()
            box.x0 = box.x0 + 0.025
            box.x1 = box.x1 + 0.025
            axs[0].set_position(box)
        
    def plot_comparison(
        self,
        img1=None,
        img2=None,
        inset=[0.7,-0.2,0.7,0.7],
        extent=[550,1800,2800,3650],
        targ=(993,3397),
        match1=(920,3480),
        match2=(1073,3521),
        clip_x=3800,
        cbar_inset=[-0.15,0.1,0.02,0.8],
        fontsize=22,
        include=["euclidian","geodesic"],
        csv_file=None,
        save_as=None,
        cmap="viridis",
        wspace=0.6,
        figsize=(24,5),
        leg_anchor=(3.6,1)):

        # set inputs
        self.img1 = img1
        self.img2 = img2
        self.inset = inset
        self.extent = extent
        self.targ = targ
        self.match1 = match1
        self.match2 = match2
        self.clip_x = clip_x
        self.cbar_inset = cbar_inset
        self.include = include
        self.csv_file = csv_file
        self.cmap = cmap
        self.wspace = wspace
        self.figsize = figsize
        self.leg_anchor = leg_anchor

        # initialize figure
        self.fig,self.axs = plt.subplots(
            ncols=4, 
            figsize=self.figsize,
            gridspec_kw={
                "wspace": self.wspace,
                "width_ratios": [0.6,0.6]+[0.05 for ii in range(len(self.include))]
            })

        # plot the unsmoothed version
        self.plot_surfaces(
            img_dist=self.img1,
            inset_axis=self.inset,
            inset_extent=self.extent,
            clip_x=self.clip_x,
            coord_targ=self.targ,
            coord_closest=self.match1,
            axs=self.axs[0],
            cbar_inset=None
        )

        colors = sns.color_palette(self.cmap,2)
        self.axs[0].set_title(
            "unsmoothed",
            fontname=self.plot_defaults.fontname,
            fontsize=fontsize,
            color=colors[0],
            fontweight="bold")

        # plot smoothed version
        self.plot_surfaces(
            img_dist=self.img2,
            inset_axis=self.inset,
            inset_extent=self.extent,
            clip_x=self.clip_x,
            coord_targ=self.targ,
            coord_closest=self.match2,
            axs=self.axs[1],
            cbar_inset=None
        )

        self.axs[1].set_title(
            "smoothed",
            fontname=self.plot_defaults.fontname,
            fontsize=fontsize,
            color=colors[1],
            fontweight="bold")  

        if isinstance(self.cbar_inset, list):
            cbar_ax = self.axs[0].inset_axes(self.cbar_inset)
            plotting.LazyColorbar(
                cbar_ax,
                cmap="magma_r",
                txt="distance to target [dva]",
                vmin=0,
                vmax=4,
                ori="vertical",
                flip_label=True
            )

        # plot distances
        self.plot_bar_comparison(
            axs=self.axs[-len(self.include):],
            csv_file=self.csv_file,
            include=self.include,
            cmap=self.cmap,
            move_box=True,
            leg_anchor=self.leg_anchor)

        plt.tight_layout()

        plotting.fig_annot(
            self.fig,
            x0_corr=-0.8,
            x_corr=-0.8)
                
        # save figure?
        if isinstance(save_as, str):
            for ext in ["png","svg"]:

                fname = f"{save_as}.{ext}"
                utils.verbose(f"Writing '{fname}'", self.verbose)

                self.fig.savefig(
                    fname,
                    bbox_inches="tight",
                    dpi=300,
                    facecolor="white"
                )

        plt.show()            

class WholeBrainToLine(MainFigure):

    def __init__(
        self,
        TR=0.105,
        hrf_pars=[1,4.6,0],
        h5_file=None,
        **kwargs):

        self.h5_file = h5_file
        self.TR = TR
        self.hrf_pars = hrf_pars
        self.__dict__.update(kwargs)
        super().__init__(**kwargs)

        self.read_attributes = [
            'df_exp_r2', 
            'df_exp_pred',
            'df_exp_pars']

        if isinstance(self.h5_file, str) and os.path.exists(self.h5_file):
            self.from_hdf(self.h5_file)
        else:
            self.parallel_predictions()
            self.to_hdf(output_file=self.h5_file)

    def from_hdf(self, input_file=None):

        if input_file == None:
            raise ValueError("No output file specified")
        else:
            self.h5_file = input_file

        utils.verbose(f"Reading from {self.h5_file}", self.verbose)
        hdf_store = pd.HDFStore(self.h5_file)
        hdf_keys = hdf_store.keys()
        for key in hdf_keys:
            key = key.strip("/")
            
            if self.verbose:
                utils.verbose(f" Setting attribute: {key}", self.verbose)

            setattr(self, key, hdf_store.get(key))

        utils.verbose("Done", self.verbose)
        hdf_store.close()     

    def to_hdf(self, output_file=None, overwrite=False):

        if output_file == None:
            raise ValueError("No output file specified")
        else:
            self.h5_file = output_file

        if overwrite:
            if os.path.exists(self.h5_file):
                store = pd.HDFStore(self.h5_file)
                store.close()
                os.remove(self.h5_file)

        utils.verbose(f"Saving to {self.h5_file}", self.verbose)
        for attr in self.read_attributes:
            if hasattr(self, attr):
                
                if self.verbose:
                    utils.verbose(f" Storing attribute: {attr}", self.verbose)
                    
                add_df = getattr(self, attr)
                if os.path.exists(self.h5_file):
                    add_df.to_hdf(self.h5_file, key=attr, append=True, mode='r+', format='t')
                else:
                    store = pd.HDFStore(self.h5_file)
                    store.close()
                    add_df.to_hdf(self.h5_file, key=attr, mode='w', format='t')
        
        utils.verbose("Done", self.verbose)

        store = pd.HDFStore(self.h5_file)
        store.close()      

    def parallel_predictions(self, parallel=True):
        
        if parallel:
            n_jobs = len(self.process_subjs)
            self.parallel_output = Parallel(n_jobs=n_jobs,verbose=False)(
                delayed(self.fetch_predictions)(subject)
                for subject in self.process_subjs
            )
        else:
            self.parallel_output = []
            for subject in ["sub-001"]:
                self.parallel_output.append(self.fetch_predictions(subject))

        # parse output to full dataframes
        self.df_exp_r2 = []
        self.df_exp_pred = []
        self.df_exp_pars = []
        for ix in range(len(self.parallel_output)):
            self.df_exp_pred.append(self.parallel_output[ix][0])
            self.df_exp_r2.append(self.parallel_output[ix][1])
            self.df_exp_pars.append(self.parallel_output[ix][2])

        self.df_exp_pred = pd.concat(self.df_exp_pred)
        self.df_exp_r2 = pd.concat(self.df_exp_r2)
        self.df_exp_pars = pd.concat(self.df_exp_pars)

    def fetch_predictions(
        self,
        subject=None):
        
        utils.verbose(f"Dealing with {subject}", self.verbose)

        # initialize r2 dataframe
        self.df_subj_r2 = {}
        for param in ["acq","code","r2"]:
            self.df_subj_r2[param] = []

        # get design matrix and functional data
        ses = self.subj_obj.get_session(subject)
        dm = opj(
            self.deriv,
            "prf",
            subject,
            f"ses-{ses}",
            f"{subject}_ses-{ses}_task-pRF_run-avg_desc-design_matrix.mat"
            )

        func = opj(
            self.deriv,
            "prf",
            subject,
            f"ses-{ses}",
            f"{subject}_ses-{ses}_task-pRF_run-avg_vox-avg_desc-data.npy"
            )            

        # initialize prf objects
        self.sb = self.initialize_prf_objects(
            subject=subject,
            func=func,
            dm=dm)

        # get design matrix as block design
        dm_as_block = np.array([self.sb["line_obj"].design_matrix[...,ii].sum()>0 for ii in range(self.sb["line_obj"].design_matrix.shape[-1])])

        # convolve with HRF
        avg_tc = np.squeeze(self.sb["tc_bold"])
        pred_block = self.run_glm(
            avg_tc,
            dm_as_block,
            convolve_hrf=True,
            hrf_pars=self.hrf_pars
        )

        # get r2
        r2_block = r2_score(avg_tc,pred_block)
        # utils.verbose(f" r2 of GLM with pRF design as block design: {r2_block}", self.verbose)
        for ii,par in zip(list(self.df_subj_r2.keys()),["block",0,r2_block]):
            self.df_subj_r2[ii].append(par)

        # run GLM on wb prediction
        pred_wb_fitted = self.run_glm(
            avg_tc,
            self.sb["wb_pred"],
            convolve_hrf=False
        )

        # get r2
        r2_wb = r2_score(avg_tc,pred_wb_fitted)
        # utils.verbose(f" r2 of GLM with on wb prediction: {r2_wb}", self.verbose)
        for ii,par in zip(list(self.df_subj_r2.keys()),["wb",1,r2_wb]):
            self.df_subj_r2[ii].append(par)

        for ii,par in zip(list(self.df_subj_r2.keys()),["line",2,getattr(self.sb["line_obj"], f"{self.model}_iter")[0,-1]]):
            self.df_subj_r2[ii].append(par)                

        self.df_subj_r2 = pd.DataFrame(self.df_subj_r2)
        self.df_subj_r2["subject"] = subject

        # put preditions in dataframe
        self.df_subj_pred = {}
        for ii,par in zip(
            ["wb_pred_glm","line_pred","wb_block","wb_pred","tc_bold"],
            [pred_wb_fitted,self.sb["line_pred"],pred_block,self.sb["wb_pred"],self.sb["tc_bold"]]):
            self.df_subj_pred[ii] = np.squeeze(par)

        self.df_subj_pred = pd.DataFrame(self.df_subj_pred)
        self.df_subj_pred["subject"] = subject
        self.df_subj_pred["t"] = list(np.arange(0,self.df_subj_pred.iloc[:,0].values.shape[0])*self.TR)
        self.df_subj_pred = self.df_subj_pred.set_index(["subject","t"])

        # get predicted parameters
        self.df = []
        for ix,ii in enumerate(["line_pars","wb_pars"]):
            self.tmp = prf.Parameters(self.sb[ii], model=self.model).to_df()
            self.tmp["type"] = ii
            self.tmp["code"] = ix
            self.tmp["subject"] = subject
            self.df.append(self.tmp)

        self.df_subj_pars = pd.concat(self.df)
        self.df_subj_pars = self.df_subj_pars.set_index(["subject","type","code"])

        return self.df_subj_pred,self.df_subj_r2,self.df_subj_pars

    def run_glm(
        self,
        data,
        regressor,
        convolve_hrf=False,
        **kwargs):
        
        if convolve_hrf:
            self.hrf = glm.define_hrf(**kwargs)
            self.stim_vector = glm.convolve_hrf(
                self.hrf, 
                regressor, 
                TR=self.TR,
                make_figure=False)
        else:
            self.stim_vector = regressor.copy()

        # create design matrix
        self.lvl1 = glm.first_level_matrix(self.stim_vector)

        # run glm
        self.results = glm.fit_first_level(
            self.lvl1, 
            data, 
            make_figure=False)
        
        self.betas = self.results["betas"][1,:]
        self.event = self.results["x_conv"][:,1][...,np.newaxis]
        
        return np.squeeze(self.event@self.betas)

    def plot_block_vs_wb(
        self, 
        axs=None, 
        exclude_line=True, 
        posthoc=False, 
        **kwargs):

        if axs == None:
            if exclude_line:
                figsize = (1.5,8)
            else:
                figsize = (2,8)

            _,axs = plt.subplots(figsize=figsize)
        
        # exclude line from bar plot or not (default = True)
        if exclude_line:
            df = utils.select_from_df(self.df_exp_r2, expression="acq != line")
        else:
            df = self.df_exp_r2.copy()

        self.block_plot = plotting.LazyBar(
            data=df,
            x="acq",
            y="r2",
            sns_ori="v",
            sns_rot=-20,
            sns_offset=5,
            axs=axs,
            add_labels=True,
            color="#cccccc",
            add_points=True,
            points_cmap=self.sub_colors,
            points_hue="subject",    
            y_label2="cvR$^2$",
            lim=[0,1],
            ticks=list(np.arange(0,1.2,0.2)),
            fancy=True,
            trim_bottom=True,
            labels=["block","wb","line"])

        # run posthoc?
        if posthoc:
            self.posth = glm.Posthoc(
                df=df,
                dv="r2",
                between="acq",
                axs=axs,
                **kwargs)

            self.posth.plot_bars()

    def plot_predictions(
        self, 
        axs=None,
        subject="sub-002",
        figsize=(24,5),
        vf_extent=[-5,5],
        save_as=None,
        **kwargs):
        
        # set bar width according to number of bars
        if len(kwargs) > 0:
            if "exclude_line" in list(kwargs.keys()):
                ww = 0.15
            else:
                ww = 0.08

        if not isinstance(axs, list):
            self.fig,(ax1,ax2,ax3) = plt.subplots(
                ncols=3,
                figsize=figsize,
                gridspec_kw={
                    "width_ratios": [1,0.5,ww]
                })

        if not hasattr(self, "color"):
            self.color = ["#cccccc","#1B9E77","#D95F02"]

        # get subject object
        subjs = np.unique(self.df_exp_pred.reset_index()["subject"])
        if not subject in list(subjs):
            subj = list(subjs)[0]
        else:
            subj = subject
        
        self.data_for_plot = utils.select_from_df(self.df_exp_pred, expression=f"subject = {subj}")
        self.pars_for_plot = utils.select_from_df(self.df_exp_pars, expression=f"subject = {subj}")

        self.x_axis = np.array(list(np.arange(0,self.data_for_plot.shape[0])*self.TR))
        plotting.LazyPlot(
            [self.data_for_plot[qq].values for qq in ["tc_bold","line_pred","wb_pred_glm"]],
            xx=self.x_axis,
            add_hline="default",
            color=self.color,
            line_width=[1,3,3],
            markers=['.',None,None],
            x_label="time (s)",
            y_label="amplitude (%)",
            labels=['data','line','whole brain'],
            axs=ax1,
            x_lim=[0,int(self.x_axis[-1])],
            x_ticks=list(np.arange(0,self.x_axis[-1]+40,40)),
            y_lim=[-1,1.5],
            y_ticks=[-1,-0.5,0,0.5,1,1.5])

        # make prfs
        plotting.LazyPRF(
            np.zeros((500,500)), 
            cmap=utils.make_binary_cm(self.color[-1]),
            vf_extent=vf_extent,
            cross_color="k",
            edge_color=None,
            shrink_factor=0.9,
            vf_only=True,
            ax=ax2)

        for ii,col in zip([0,1],self.color[-2:]):

            pars = utils.select_from_df(self.pars_for_plot, expression=f"code = {ii}").values
            circ = plt.Circle(
                (pars[0,0],pars[0,1]),
                pars[0,2],
                ec=col,
                lw=2,
                fill=False)

            ax2.add_artist(circ)

        for ii,val in zip(["-5°","5°","-5°","5°"], [(0,0.51),(0.98,0.51),(0.51,0),(0.51,0.96)]):
            ax2.annotate(
                ii,
                val,
                fontsize=self.plot_defaults.label_size,
                xycoords="axes fraction"
            )

        # bar plot with wb pRF vs design as block
        self.plot_block_vs_wb(axs=ax3, **kwargs)

        # move the timecourse plot towards middle
        plt.tight_layout()

        box = ax1.get_position()
        box.x0 += 0.05
        box.x1 += 0.05
        ax1.set_position(box)

        # add annotations
        plotting.fig_annot(
            self.fig,
            x0_corr=-1,
            x_corr=-0.8)

        # save figure?
        if isinstance(save_as, str):
            for ext in ["png","svg"]:

                fname = f"{save_as}.{ext}"
                utils.verbose(f"Writing '{fname}'", self.verbose)

                self.fig.savefig(
                    fname,
                    bbox_inches="tight",
                    dpi=300,
                    facecolor="white"
                )

    def initialize_prf_objects(
        self, 
        subject=None,
        func=None,
        dm=None):

        # initiate class
        output = {}
        for ix,it in zip(["wb","line"],[0,2]):
            # utils.verbose(f" Initializing '{ix}' pRF-object", self.verbose)
            tmp = prf.pRFmodelFitting(
                func,
                design_matrix=dm,
                TR=self.TR,
                verbose=False,
                screen_distance_cm=196
                )

            # load existing parameters based on `code`
            tmp_pars = utils.select_from_df(self.df_params, expression=(f"subject = {subject}", "&", f"code = {it}"))

            tmp.load_params(
                tmp_pars,
                model=self.model,
                stage="iter")

            pars,rf,tc,pred = tmp.plot_vox(model=self.model,make_figure=False)

            output[f"{ix}_pars"] = pars
            output[f"{ix}_obj"] = tmp
            output[f"{ix}_rf"] = rf
            output[f"{ix}_pred"] = pred
        
        output["tc_bold"] = tc

        return output

class AnatomicalPrecision(MainFigure):

    def __init__(
        self,
        reg_csv=None,
        **kwargs):

        self.reg_csv = reg_csv
        self.__dict__.update(kwargs)
        super().__init__(**kwargs)

        if isinstance(self.reg_csv, str):
            self.df_reg = pd.read_csv(self.reg_csv)
    
    def plot_individual_distributions(
        self, 
        axs=None,
        add_title=True):
        
        if not isinstance(axs, (list,np.ndarray)):
            _,axs = plt.subplots(ncols=len(self.process_subjs), figsize=(24,5))
        else:
            if len(axs) != len(self.process_subjs):
                raise ValueError(f"Number of axes ({len(axs)}) must match number of subjects ({len(self.process_subjs)})")
                  

        self.fwhm_objs = []
        for sub_ix,subject in enumerate(self.process_subjs):
            ax = axs[sub_ix]
            y_lbl = None
            if sub_ix == 0:
                y_lbl = "count"

            y_data = utils.select_from_df(self.df_reg, expression=f"subject = {subject}")['euclidian'].values
            self.reg_plot = plotting.LazyHist(
                y_data,
                axs=ax,
                kde=True,
                hist=True,
                fill=False,
                y_label2=y_lbl,
                x_label2="distance (mm)",
                color=self.sub_colors[sub_ix],
                hist_kwargs={"alpha": 0.4},
                kde_kwargs={"linewidth": 4},
            )

            if add_title:
                ax.set_title(
                    subject, 
                    fontsize=self.reg_plot.font_size, 
                    color=self.sub_colors[sub_ix], 
                    fontweight="bold")            
            
            # get kde line
            self.fwhm_objs.append(fitting.FWHM(self.reg_plot.kde[0],self.reg_plot.kde[-1]))

    def plot_spread_as_bar(self, axs=None):
        
        if axs == None:
            _,axs = plt.subplots(figsize=(4,8))    

        plotting.LazyBar(
            data=self.df_reg,
            x="subject",
            y="euclidian",
            cmap=self.cmap,
            axs=axs,
            sns_ori="v",
            x_label2="subjects",
            y_label2="registration variation (mm)",
            fancy=True,
            sns_offset=5,
            add_points=True,
            lim=[0,0.5],
            points_alpha=0.5,
            points_color="#cccccc",
            trim_bottom=True
        )

    def plot_fwhm_as_bar(self, axs=None):
        
        if axs == None:
            _,axs = plt.subplots(figsize=(1,8))

        self.y_fwhm = [i.fwhm for i in self.fwhm_objs]
        self.df_fwhm = pd.DataFrame(self.y_fwhm, columns=["fwhm"])
        self.df_fwhm["subject"], self.df_fwhm["ix"] = self.process_subjs, 0

        plotting.LazyBar(
            data=self.df_fwhm,
            x="ix",
            y="fwhm",
            sns_ori="v",
            axs=axs,
            sns_offset=4,
            color="#cccccc",
            add_points=True,
            points_cmap=self.cmap,
            points_hue="subject",    
            y_label2="euclidian spread (mm)",
            fancy=True,
            lim=[0,0.1],
            trim_bottom=True
        )
    
    def plot_subject_beam(
        self, 
        subject=None, 
        axs=None,
        inset_axis=[0.6,-0.4,0.7,0.7]):
        
        if axs == None:
            _,axs = plt.subplots(figsize=(8,8))

        ses = self.subj_obj.get_session(subject)
        inset_extent = self.subj_obj.get_extent(subject)
        img_beam = opj(self.results_dir, subject, f"{subject}_ses-{ses}_desc-slice_on_surf.png")

        if not os.path.exists(img_beam):
            raise FileNotFoundError(f"Could not find file '{img_beam}'. Please create it with pRFline/scripts/slice_on_surf.py")

        img_d = imageio.v2.imread(img_beam)
        axs.imshow(img_d)
        x1, x2, y1, y2 = inset_extent
        axs.set_xlim(x1,x2)
        axs.set_ylim(y2,y1)
        axs.set_xticklabels([])
        axs.set_yticklabels([])    
        axs.axis('off')

        # inset axis
        ax2 = axs.inset_axes(inset_axis)
        ax2.imshow(img_d)

        # fix connecting lines
        self.rect,self.lines = ax2.indicate_inset_zoom(axs, ec="#cccccc")

        coords = self.rect.get_patch_transform().transform(self.rect.get_path().vertices[:-1])
        self.lines[0].xy2 = coords[0]
        self.lines[3].xy2 = coords[2]

        for ii in self.lines:
            ii.set_facecolor("#cccccc")
            ii.set_linewidth(0.5)
        
        self.rect.set_edgecolor("#cccccc")

        ax2.axis('off')

    def plot_beam_on_surface(
        self, 
        axs=None,
        add_title=True):

        if not isinstance(axs, (list,np.ndarray)):
            _,axs = plt.subplots(ncols=len(self.process_subjs), figsize=(24,5))
        else:
            if len(axs) != len(self.process_subjs):
                raise ValueError(f"Number of axes ({len(axs)}) must match number of subjects ({len(self.process_subjs)})")

        for sub_ix,subject in enumerate(self.process_subjs):
            ax = axs[sub_ix]

            self.plot_subject_beam(
                axs=ax,
                subject=subject)

            if add_title:
                ax.set_title(
                    subject, 
                    fontsize=24, 
                    color=self.sub_colors[sub_ix], 
                    fontweight="bold")                

    def plot_smoothed_target(
        self, 
        subject="sub-003", 
        axs=None,
        inset_axis=[1.2,-3,3,3],
        inset_extent=[500,900,1300,1700]):
        
        if axs == None:
            _,axs = plt.subplots(figsize=(2,2))

        ses = self.subj_obj.get_session(subject)
        img_beam = opj(self.results_dir, subject, f"{subject}_ses-{ses}_desc-target_on_surf.png")

        if not os.path.exists(img_beam):
            raise FileNotFoundError(f"Could not find file '{img_beam}'. Please create it with pRFline/scripts/slice_on_surf.py")

        img_d = imageio.v2.imread(img_beam)
        axs.imshow(img_d)
        axs.set_xticklabels([])
        axs.set_yticklabels([])    
        axs.axis('off')

        # inset axis
        x1, x2, y1, y2 = inset_extent
        ax2 = axs.inset_axes(inset_axis)
        ax2.set_xlim(x1,x2)
        ax2.set_ylim(y2,y1)
        ax2.imshow(img_d)

        # fix connecting lines
        self.rect,self.lines = axs.indicate_inset_zoom(ax2, ec="#cccccc")

        coords = self.rect.get_patch_transform().transform(self.rect.get_path().vertices[:-1])
        self.lines[0].xy2 = coords[0]
        self.lines[3].xy2 = coords[2]

        for ii in self.lines:
            ii.set_facecolor("#cccccc")
            ii.set_linewidth(0.5)
        
        self.rect.set_edgecolor("#cccccc")

        ax2.axis('off')

    def plot_reg_overview(
        self, 
        axs=None,
        fc=[0,-0.02,-0.1,-0.18,-0.26]):

        if not isinstance(axs, (list,np.ndarray)):
            fig,axs = plt.subplots(ncols=5, figsize=(24,5))
        else:
            if len(axs) != 5:
                raise ValueError(f"Number of axes ({len(axs)}) must be 5")

        self.reg_imgs = []
        for ii,tt in enumerate(["high-res","low-res","partial FOV","slice","line"]):
            
            ax = axs[ii]
            img = opj(self.results_dir, "figure_parts", f"anat_0{ii+1}.png")
            img_d = imageio.v2.imread(img)

            box = ax.get_position()
            box.x0 += fc[ii]
            box.x1 += fc[ii] 

            ax.set_position(box)
            ax.imshow(img_d)
            ax.annotate(
                tt, 
                (0.5,0.85), 
                ha="center",
                fontsize=20, 
                xycoords="axes fraction")

            ax.axis('off')

        xyA = [1150,800]
        xyB = [250,800]
        # ConnectionPatch handles the transform internally so no need to get fig.transFigure
        arrow = patches.ConnectionPatch(
            xyA,
            xyB,
            coordsA=axs[0].transData,
            coordsB=axs[1].transData,
            color="#cccccc",
            arrowstyle="-| >",  # "normal" arrow
            mutation_scale=25,  # controls arrow head size
            linewidth=1,
        )

        try:
            fig.patches.append(arrow)
            axs[1].annotate(
                "ANTs", 
                (-0.195,0.46),
                color="#cccccc",
                fontweight="bold",
                fontsize=18, 
                xycoords="axes fraction")
        except:
            pass

    def compile_reg_figure(
        self,
        save_as=None,
        fontsize=28,
        **kwargs):

        utils.verbose("Compiling figure. May take a few minutes..", True)

        # initialize figure
        self.fig = plt.figure(figsize=(24,13))
        self.sf0 = self.fig.subfigures(
            nrows=3, 
            height_ratios=[0.7,0.6,0.5],
            hspace=0.2)

        # do this sketchy bit to get one tiny axis next to a big one
        self.gsbig = self.sf0[0].add_gridspec(
            ncols=6, 
            width_ratios=[0.5,1,1,1,1,1], 
            wspace=0.1)

        self.axs = []
        for ii in range(5):
            self.axs.append(self.sf0[0].add_subplot(self.gsbig[1+ii]))

        self.gssmall = self.sf0[0].add_gridspec(ncols=2, nrows=3, width_ratios=[0.1,0.75])
        ax2 = self.sf0[0].add_subplot(self.gssmall[0])

        self.sf0_ax1 = self.sf0[1].subplots(ncols=len(self.process_subjs), gridspec_kw={"wspace": 0.2})
        self.sf0_ax2 = self.sf0[2].subplots(ncols=len(self.process_subjs), gridspec_kw={"wspace": 0.2})

        plt.tight_layout()
        self.plot_reg_overview(
            axs=self.axs,
            fc=[0.195,0.18,0.12,0.06,0])

        # add the individual elements to their axes
        self.plot_smoothed_target(axs=ax2, **kwargs)
        self.plot_beam_on_surface(axs=self.sf0_ax1)
        self.plot_individual_distributions(
            axs=self.sf0_ax2, 
            add_title=False)

        # make the arrow
        xyA = [1150,800]
        xyB = [250,800]
        arrow = patches.ConnectionPatch(
            xyA,
            xyB,
            coordsA=self.axs[0].transData,
            coordsB=self.axs[1].transData,
            color="#cccccc",
            arrowstyle="-| >",  # "normal" arrow
            mutation_scale=25,  # controls arrow head size
            linewidth=1,
        )

        self.fig.add_artist(arrow)
        self.axs[1].annotate(
            "ANTs", 
            (-0.15,0.46),
            color="#cccccc",
            fontweight="bold",
            fontsize=28, 
            xycoords="axes fraction")

        # make annotations
        top_y = 1.02
        for y_pos,let,ax in zip(
            [top_y,0.6,0.31],
            ["A","C","D"],
            [ax2,self.sf0_ax1[0],self.sf0_ax2[0]]):

            ax.annotate(
                let, 
                (0,y_pos), 
                fontsize=fontsize, 
                xycoords="figure fraction")

        y = 0.98
        for ax,ses,x in zip(
            [self.axs[0],self.axs[2]],
            ["ses-1","ses-2"], 
            [0.4,0.75]):

            ax.text(
                *(x,y), 
                ses,
                size=26,
                # bbox=dict(boxstyle="round",
                #     ec=(1.,0.5,0.5),
                #     fc=(1.,0.8,0.8)),
                transform=ax.transAxes)

        # draw lines below ses-X
        for ix,ax in enumerate(self.axs):
            if ix == 0:
                xx = [0.2,0.8]
            elif ix == 1:
                xx = [0.2,1]
            elif ix == len(self.axs)-1:
                xx = [0,0.8]
            else:
                xx = [0,1]

            line = lines.Line2D(
                xx, 
                [0.95,0.95], 
                lw=1, 
                color='k', 
                transform=ax.transAxes)

            ax.add_artist(line)

        # panel B is slightly more annoying
        bbox = self.axs[0].get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        x_pos = self.axs[0].get_position().x0 + (bbox.width*0.005)
        self.axs[0].annotate(
            "B", 
            (x_pos,top_y), 
            fontsize=fontsize, 
            xycoords="figure fraction")        

        # save figure
        if isinstance(save_as, str):
            for ext in ["png","svg"]:

                fname = f"{save_as}.{ext}"
                utils.verbose(f"Writing '{fname}'", self.verbose)

                self.fig.savefig(
                    fname,
                    bbox_inches="tight",
                    dpi=300,
                    facecolor="white"
                )


class DepthHRF(MainFigure, prf.pRFmodelFitting):

    def __init__(
        self,
        subject="sub-003",
        colors=["#FF0000","#0000FF"],
        hrf_csv=None,
        metric_csv=None,
        hrf_length=30,
        code=3,
        **kwargs):
        
        self.subject = subject
        self.colors = colors
        self.hrf_csv = hrf_csv
        self.metric_csv = metric_csv
        self.hrf_length = hrf_length
        self.code = code
        self.__dict__.update(kwargs)
        MainFigure.__init__(self, **kwargs)

        # get info we need
        self.get_information()

        if isinstance(self.hrf_csv, str) and os.path.exists(self.hrf_csv):
            utils.verbose(f"Reading '{self.hrf_csv}'", self.verbose)
            self.hrf_df = pd.read_csv(self.hrf_csv).set_index(["subject","t","depth"])
        else:
            self.fetch_hrfs_across_depth(hrf_length=self.hrf_length, code=self.code)

            if not isinstance(self.hrf_csv, str):
                self.hrf_csv = opj(self.data_dir, f"sub-all_model-{self.model}_desc-hrf_across_depth.csv")

            utils.verbose(f"Writing '{self.hrf_csv}'", self.verbose)
            self.hrf_df.to_csv(self.hrf_csv)

        # fetch the metrics
        if isinstance(self.metric_csv, str) and os.path.exists(self.metric_csv):
            utils.verbose(f"Reading '{self.metric_csv}'", self.verbose)
            self.df_hrf_metrics = pd.read_csv(self.metric_csv).set_index(["subject","level","depth"])
        else:
            self.metric_csv = opj(self.data_dir, f"sub-all_model-{self.model}_desc-hrf_metrics.csv")

            # fetch metrics from self.hrf_df
            self.fetch_all_hrf_metrics()

            utils.verbose(f"Writing '{self.metric_csv}'", self.verbose)
            self.df_hrf_metrics.to_csv(self.metric_csv)
        
    def fetch_single_hrf_metrics(self, subject=None):
        
        # initalize output
        df_mag = {}
        for ii in ["subject","level","depth","mag","fwhm","ttp"]:
            df_mag[ii] = []

        # get subject specific hrfs
        sub_hrf = utils.select_from_df(self.hrf_df, expression=f"subject = {subject}")

        # parse them into list depending on the number of voxels in estimates
        depths = np.unique(sub_hrf.reset_index()["depth"].values)
        depths_pc = np.arange(0,depths.shape[0])/(depths.shape[0]-1)*100            

        hrf_list = []
        for ii in depths:
            df_depth = utils.select_from_df(sub_hrf, expression=f"depth = {ii}")
            dd = df_depth['hrf'].values
            
            if not np.isnan(dd.sum()) or not np.all(dd==0):
                hrf_list.append(dd)                

                df_mag["subject"].append(subject)
                df_mag["depth"].append(depths_pc[ii])
                df_mag["level"].append(ii)

        # create time axis
        time_axis = list(np.arange(0,hrf_list[0].shape[0])*self.TR)

        # add time-to-peak across the ribbon as inset-axis
        tcs = np.array(hrf_list)
        peak_positions = (np.argmax(tcs, axis=1)/tcs.shape[-1])*df_depth.reset_index().t.iloc[-1]

        # FWHM
        y_fwhm = np.zeros((len(hrf_list)))
        for hrf_ix,hrf in enumerate(hrf_list):
            fwhm = fitting.FWHM(time_axis, hrf).fwhm
            y_fwhm[hrf_ix] = fwhm

        # get magnitudes
        mag = np.array([np.amax(i) for i in hrf_list])

        for par,var in zip(["mag","fwhm","ttp"],[mag,y_fwhm,peak_positions]):
            for v in var:
                df_mag[par].append(v)

        return pd.DataFrame(df_mag)

    def fetch_all_hrf_metrics(self):

        self.df_hrf_metrics = []
        utils.verbose(f"Extracting HRF metrics from profiles..", self.verbose)
        n_jobs = len(self.process_subjs)
        hrf_ = Parallel(n_jobs=n_jobs,verbose=False)(
            delayed(self.fetch_single_hrf_metrics)(subject)
            for subject in self.process_subjs
        )

        self.df_hrf_metrics = pd.concat(hrf_).set_index(["subject","level","depth"])

    def plot_hrf_profiles(
        self, 
        axs=None, 
        insets="mag",
        inset_axis=[0.75, 0.65, 0.3, 0.3],
        add_title=True):

        if not isinstance(axs, (list,np.ndarray)):
            _,axs = plt.subplots(ncols=len(self.process_subjs), figsize=(24,5))
        else:
            if len(axs) != len(self.process_subjs):
                raise ValueError(f"Number of axes ({len(axs)}) must match number of subjects ({len(self.process_subjs)})")

        self.df_mag = {}
        for ii in ["subject","level","depth","mag","fwhm","ttp"]:
            self.df_mag[ii] = []

        for ix,subject in enumerate(self.process_subjs):

            # get subject-specific HRFs from HRF-dataframe  
            sub_hrf = utils.select_from_df(self.hrf_df, expression=f"subject = {subject}")
            sub_met = utils.select_from_df(self.df_hrf_metrics, expression=f"subject = {subject}")

            # parse them into list depending on the number of voxels in estimates
            depths = np.unique(sub_hrf.reset_index()["depth"].values)

            hrf_list = []
            for ii in depths:
                dd = utils.select_from_df(sub_hrf, expression=f"depth = {ii}")['hrf'].values
                
                if not np.isnan(dd.sum()) or not np.all(dd==0):
                    hrf_list.append(dd)

            # get subject specific color palette
            colors = sns.color_palette(f"light:{self.sub_colors.as_hex()[ix]}", len(hrf_list))
            y_ticks = None
            if ix == 0:
                y_lbl = "magnitude"
            else:
                y_lbl = None

            # create time axis
            time_axis = list(np.arange(0,hrf_list[0].shape[0])*self.TR)

            # plot
            y_max = np.amax(np.array(hrf_list))
            y_ticks = [0,round(y_max/2,2),round(y_max,2)]
            plotting.LazyPlot(
                hrf_list,
                axs=axs[ix],
                xx=time_axis,
                x_label="time (s)",
                y_label=y_lbl,
                cmap=colors,
                # x_lim=[0,25],
                # x_ticks=np.arange(0,30,5),
                y_ticks=y_ticks,
                add_hline=0,
                trim_left=False
            )

            # decide plot properties depending on which type to put on the inset axis
            if isinstance(insets, str):

                ax2 = axs[ix].inset_axes(inset_axis)
                if insets == "mag":
                    ori = "v"
                    y_lab = "magnitude"
                    x_lab = "depth"
                elif insets == "ttp":
                    ori = "h"
                    y_lab = "depth"
                    x_lab = "time-to-peak (s)"
                elif insets == "fwhm":
                    ori = "v"
                    y_lab = "FWHM"
                    x_lab = "depth"
                else:
                    raise ValueError(f"insets must be one of 'mag','ttp', or 'fwhm'; not '{insets}'")

                plotting.LazyBar(
                    data=sub_met.reset_index(),
                    x="level",
                    y=insets,
                    axs=ax2,
                    label_size=9,
                    font_size=12,
                    palette=colors,
                    sns_ori=ori,
                    add_labels=True,
                    y_label2=y_lab,
                    x_label2=x_lab,
                    alpha=0.8,
                    fancy=True,
                    sns_offset=3,
                    trim_bottom=True)               

            if add_title:
                axs[ix].set_title(
                    subject, 
                    fontsize=24, 
                    color=self.sub_colors[ix], 
                    fontweight="bold")

    def get_information(self):

        # get session
        ses = self.subj_obj.get_session(self.subject)

        # get functional data
        self.fn_func = opj(
            self.deriv, 
            "prf", 
            self.subject,
            f"ses-{ses}",
            f"{self.subject}_ses-{ses}_task-pRF_run-avg_vox-all_desc-data.npy")

        if not os.path.exists(self.fn_func):
            raise FileNotFoundError(f"Could not find file '{self.fn_func}'")
        else:
            self.func = np.load(self.fn_func).T

        # get design matrix
        self.fn_dm = opj(
            os.path.dirname(self.fn_func),
            f"{self.subject}_ses-{ses}_task-pRF_run-avg_desc-design_matrix.mat")

        if not os.path.exists(self.fn_dm):
            raise FileNotFoundError(f"Could not find design matrix '{self.fn_dm}'")

        # initiate object
        prf.pRFmodelFitting.__init__(
            self,
            self.func,
            design_matrix=self.fn_dm,
            TR=0.105,
            verbose=self.verbose
        )

        # load params
        self.df_pars = utils.select_from_df(self.df_params, expression=(f"subject = {self.subject}", "&", "code = 4"))
        self.pars_array = prf.Parameters(self.df_pars, model=self.model).to_array()
        self.load_params(self.pars_array, model=self.model, stage="iter")
        
        # get the predictions
        rib_range = self.subj_obj.get_ribbon(self.subject)
        self.pial_pars,_,self.pial_tc,self.pial_pred = self.plot_vox(
            vox_nr=rib_range[0],
            model=self.model,
            make_figure=False
        )

        self.wm_pars,_,self.wm_tc,self.wm_pred = self.plot_vox(
            vox_nr=(rib_range[1]-1), # account for indexing
            model=self.model,
            make_figure=False
        )

        # create HRFs
        self.pial_hrf = glm.define_hrf([1,self.df_pars.hrf_deriv.iloc[rib_range[0]],0])[0]
        self.wm_hrf = glm.define_hrf([1,self.df_pars.hrf_deriv.iloc[rib_range[1]-1],0])[0]

    def plot_pial_wm_timecourses(self, axs=None):

        if axs == None:
            _,axs = plt.subplots(figsize=(15,5))

        self.x_axis = np.array(list(np.arange(0,self.func.shape[-1])*self.TR))
        plotting.LazyPlot(
            [self.pial_tc,self.pial_pred,self.wm_tc,self.wm_pred],
            xx=self.x_axis,
            line_width=[0.5,2,0.5,2],
            color=[x for item in self.colors for x in repeat(item, 2)],
            axs=axs,
            x_label="time (s)",
            y_label="response",
            labels=[
                "pial (bold)",
                "pial (pred)",
                "wm (bold)",
                "wm (pred)"],
            x_lim=[0,int(self.x_axis[-1])],
            x_ticks=list(np.arange(0,self.x_axis[-1]+40,40)),
            label_size=16,
            add_hline=0
        )

    def plot_pial_wm_prfs(self, axs=None, vf_extent=[-5,5]):

        if axs == None:
            _,axs = plt.subplots(figsize=(8,8))

        # make prfs
        plotting.LazyPRF(
            np.zeros((500,500)), 
            cmap=utils.make_binary_cm(self.colors[-1]),
            vf_extent=vf_extent,
            cross_color="k",
            edge_color=None,
            shrink_factor=0.9,
            vf_only=True,
            ax=axs)

        for ix,pr in enumerate([self.pial_pars,self.wm_pars]):
            circ = plt.Circle(
                (pr[0],pr[1]),
                pr[2],
                ec=self.colors[ix],
                lw=2,
                fill=False)

            axs.add_artist(circ)
        
        ext_list = vf_extent+vf_extent
        for ii,val in zip(ext_list, [(0,0.51),(0.96,0.51),(0.51,0),(0.51,0.96)]):
            axs.annotate(
                f"{ii}°",
                val,
                fontsize=self.plot_defaults.label_size,
                xycoords="axes fraction"
            )
            
    def plot_wmpial_hrf(
        self,
        axs=None):
        
        if axs == None:
            _,axs = plt.subplots(figsize=(8,8))

        self.hrf_ax = np.linspace(0,40,num=self.pial_hrf.shape[0])
        plotting.LazyPlot(
            [self.pial_hrf,self.wm_hrf],
            xx=self.hrf_ax,
            axs=axs,
            color=self.colors,
            x_label="time (s)",
            y_label="response",
            labels=["pial","wm"],
            x_lim=[0,25],
            x_ticks=[0,5,10,15,20,25]
        )

    def fetch_subject_hrf(
        self, 
        subject=None, 
        code=3,
        hrf_length=30):
        
        # get session
        ses = self.subj_obj.get_session(subject)

        # get subject specific dataframe
        df_subj_pars = utils.select_from_df(self.df_params, expression=(f"subject = {subject}", "&", f"code = {code}"))

        # parse into array
        arr_subj_pars = prf.Parameters(df_subj_pars, model=self.model).to_array()

        # get voxel range if we have full line estimates
        if code == 4:
            vox_range = np.arange(*self.subj_obj.get_ribbon(subject))
            arr_subj_pars = arr_subj_pars[vox_range,:]

        # loop through range
        dm = opj(
            self.deriv,
            "prf",
            subject,
            f"ses-{ses}",
            f"{subject}_ses-{ses}_task-pRF_run-avg_desc-design_matrix.mat"
            )

        if not os.path.exists(dm):
            raise FileNotFoundError(f"Could not find design matrix '{dm}'")

        # make object
        tmp = prf.pRFmodelFitting(
            None,
            design_matrix=dm,
            TR=self.TR,
            verbose=False,
            screen_distance_cm=196
            )

        # load parameters
        tmp.load_params(
            arr_subj_pars,
            model=self.model,
            stage="iter")

        # get HRF length in TR space    
        hrf_length_tr = int(hrf_length/self.TR)
        
        # loop through voxel range
        sub_hrfs = []
        for ii in range(arr_subj_pars.shape[0]):

            # get the HRF prediction
            _,_,_,hrf = tmp.plot_vox(
                vox_nr=ii,
                model=self.model,
                make_figure=False)
            
            hrf = hrf[:hrf_length_tr,...]
            
            # create time axis
            xx = list(np.arange(0,hrf.shape[0])*self.TR)

            # create dataframe
            df_hrf = pd.DataFrame(hrf, columns=["hrf"])
            df_hrf["subject"],df_hrf["t"],df_hrf["depth"] = subject,xx,ii

            sub_hrfs.append(pd.DataFrame(df_hrf))
            
        # return subject-specific HRFs
        return pd.concat(sub_hrfs)

    def fetch_hrfs_across_depth(
        self,
        hrf_length=30,
        code=3):
        
        utils.verbose(f"Fetching HRFs across depth..", self.verbose)
        n_jobs = len(self.process_subjs)
        hrf_ = Parallel(n_jobs=n_jobs,verbose=False)(
            delayed(self.fetch_subject_hrf)(
                subject,
                hrf_length=hrf_length,
                code=code
            )
            for subject in self.process_subjs
        )

        self.hrf_df = pd.concat(hrf_).set_index(["subject","t","depth"])

    def plot_positional_stability(self, axs=None, code=3, add_title=True):

        if not isinstance(axs, (list,np.ndarray)):
            _,axs = plt.subplots(ncols=len(self.process_subjs), figsize=(24,5))
        else:
            if len(axs) != len(self.process_subjs):
                raise ValueError(f"Number of axes ({len(axs)}) must match number of subjects ({len(self.process_subjs)})")

        avg_pars = utils.select_from_df(self.df_params, expression="code = 2")
        for ix,subject in enumerate(self.process_subjs):

            # get subject specific dataframe
            df_subj_pars = utils.select_from_df(self.df_params, expression=(f"subject = {subject}", "&", f"code = {code}"))

            # get voxel range if we have full line estimates
            if code == 4:
                vox_range = np.arange(*self.subj_obj.get_ribbon(subject))
                df_subj_pars = df_subj_pars.iloc[vox_range,:]

            # make colors
            colors = sns.color_palette(f"light:{self.sub_colors.as_hex()[ix]}", df_subj_pars.shape[0])

            for vox in range(df_subj_pars.shape[0]):

                cm = utils.make_binary_cm(colors[vox])
                plotting.LazyPRF(
                    np.zeros((500,500)),
                    [-5,5],
                    ax=axs[ix],
                    cmap=cm,
                    cross_color="k",
                    edge_color=None,
                    alpha=0.3
                )

                x,y,si = df_subj_pars["x"][vox],df_subj_pars["y"][vox],df_subj_pars["prf_size"][vox]
                d_line = plt.Circle(
                    (x,y),
                    si,
                    ec=self.sub_colors[ix],
                    fill=False,
                    alpha=0.4,
                    lw=2)

                axs[ix].add_artist(d_line)

            x,y,si = avg_pars["x"][ix],avg_pars["y"][ix],avg_pars["prf_size"][ix]
            sub_line = plt.Circle(
                (x,y),
                si,
                ec="k", #sub_colors[ix],
                fill=False,
                lw=2)

            axs[ix].add_artist(sub_line)

            if add_title:
                axs[ix].set_title(
                    subject, 
                    fontsize=24, 
                    color=self.sub_colors[ix], 
                    fontweight="bold",
                    y=1.02)

        for ii,val in zip(["-5°","5°","-5°","5°"], [(0,0.51),(0.96,0.51),(0.51,0),(0.51,0.96)]):
            axs[0].annotate(
                ii,
                val,
                fontsize=self.plot_defaults.label_size,
                xycoords="axes fraction"
            )                    

    def compile_depth_figure(
        self,
        figsize=(24,15),
        insets="fwhm",
        save_as=None):

        # initiate full figure
        self.fig = plt.figure(figsize=figsize, constrained_layout="tight")
        self.subfigs = self.fig.subfigures(
            nrows=3,
            height_ratios=[0.4,0.35,0.25])

        self.row1 = self.subfigs[0].subplots(
            ncols=2, 
            gridspec_kw={
                "width_ratios": [0.8,0.2], 
                "wspace": 0})

        self.row2 = self.subfigs[1].subplots(ncols=len(self.process_subjs))
        self.row3 = self.subfigs[2].subplots(ncols=len(self.process_subjs))

        # row 1 - pial/wm timecourses + their pRFs
        self.plot_pial_wm_timecourses(axs=self.row1[0])
        self.plot_pial_wm_prfs(axs=self.row1[1])

        # row 2 - response profiles
        self.plot_hrf_profiles(insets=insets, axs=self.row2, add_title=True)

        # row 3 - positional stability
        self.plot_positional_stability(axs=self.row3, add_title=False)
        
        # make annotations
        top_y = 0.97
        x_pos = 0.032
        for y_pos,let,ax in zip(
            [top_y,0.58,0.23],
            ["A","C","D"],
            [self.row1[0],self.row2[0],self.row3[0]]):

            ax.annotate(
                let, 
                (x_pos,y_pos), 
                fontsize=28, 
                xycoords="figure fraction")

        # panel B is slightly more annoying
        bbox = self.row1[1].get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        x_pos = self.row1[1].get_position().x0 + (bbox.width*0.015)
        self.row1[1].annotate(
            "B", 
            (x_pos,top_y), 
            fontsize=28, 
            xycoords="figure fraction")

        if isinstance(save_as, str):
            for ext in ["png","svg"]:

                fname = f"{save_as}.{ext}"
                utils.verbose(f"Writing '{fname}'", self.verbose)

                self.fig.savefig(
                    fname,
                    bbox_inches="tight",
                    dpi=300,
                    facecolor="white"
                )
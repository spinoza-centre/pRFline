from linescanning import(
    plotting,
    utils,
    prf,
    glm
)
from sklearn.metrics import r2_score
from pRFline.utils import (
    SubjectsDict,
    read_subject_data,
    sort_posthoc)
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import seaborn as sns
import string

opj = os.path.join

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
        if isinstance(self.full_dict, str):
            utils.verbose(f"Reading '{self.full_dict}'", self.verbose)
            if self.full_dict.endswith("csv"):
                self.df_params = pd.read_csv(self.full_dict).set_index(["subject","acq","code","run"])
        elif not isinstance(self.full_dict, dict):
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

            avg_obj = self.get_epi_avg(subject=sub)
            run_obj = self.get_epi_runs(subject=sub)
            line_obj = self.get_line_avg(subject=sub)
            rib_obj = self.get_line_ribbon(subject=sub)
            full_obj = self.get_line_all(subject=sub)

            # parse into dataframe
            epi_avg = getattr(avg_obj, f"{self.model}_iter")[self.subj_obj.get_target(sub),:]
            epi_df = prf.Parameters.to_df(epi_avg, model=self.model)
            epi_df["subject"],epi_df["run"],epi_df["acq"], epi_df["code"] = sub, "avg", "epi avg", 0

            self.df_params.append(epi_df)

            # add HRF parameters if they don't exist; ensure same shapes across EPI/lines
            for ii in ["hrf_deriv", "hrf_disp"]:
                if ii not in list(epi_df.keys()):
                    epi_df[ii] = np.nan

            for run in range(run_obj.data.shape[0]):
                pars,_,_,_ = run_obj.plot_vox(
                    vox_nr=run,
                    model=run_obj.model,
                    stage='iter',
                    make_figure=False
                )

                pars_df = prf.Parameters.to_df(pars, model=self.model)
                pars_df["subject"],pars_df["run"],pars_df["acq"],pars_df["code"] = sub, run+1,"epi runs",1

                self.df_params.append(pars_df)

            # do lines
            avg_line,_,_,_ = line_obj.plot_vox(
                model=line_obj.model,
                stage='iter',
                make_figure=False
            )

            line_df = prf.Parameters.to_df(avg_line, model=self.model)
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

                pars_df = prf.Parameters.to_df(pars, model=self.model)
                pars_df["subject"],pars_df["run"],pars_df["acq"],pars_df["code"] = sub, f"depth-{depth+1}","ribbon",3

                self.df_params.append(pars_df)

            # do full line
            pars = getattr(full_obj, f"{self.model}_iter").copy()
            vox_range = np.arange(pars.shape[0])
            pars_df = prf.Parameters.to_df(pars, model=self.model)
            pars_df["subject"],pars_df["run"],pars_df["acq"],pars_df["code"] = sub, [f"vox-{ii}" for ii in vox_range],"full",4

            self.df_params.append(pars_df)

        self.df_params = pd.concat(self.df_params).set_index(["subject","acq","code","run"])

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
            self.posth = Posthoc(
                df=self.df_r2,
                dv="r2",
                between="acq",
                axs=axs)

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
            self.df_surf = utils.select_from_df(self.df_surf, expression="subject = sub-009")

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

            # get EPI average
            avg_epi = utils.select_from_df(sub_pars, expression="code = 0").iloc[0,:].values

            # loop through runs
            for run in range(run_pars.shape[0]):

                pars = run_pars.iloc[run,:].values
                run_norm = prf.normalize_prf(avg_epi,pars)

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
            
            # extract average EPI/line-scanning pRFs
            avg_epi = utils.select_from_df(sub_pars, expression="code = 0").iloc[0,:].values
            avg_line = utils.select_from_df(sub_pars, expression="code = 2").iloc[0,:].values

            # normalize
            line_norm = prf.normalize_prf(avg_epi,avg_line)
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

            # EPI pRF
            avg_epi = utils.select_from_df(sub_pars, expression="code = 0").iloc[0,:].values
            sub_epi = plt.Circle(
                (avg_epi[0],avg_epi[1]),
                avg_epi[2],
                ec="k",
                fill=False,
                lw=2)

            axs[ix].add_artist(sub_epi)    

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


    def get_epi_runs(self, subject=None):
        return self.full_dict[subject]['wb']['runs']

    def get_epi_avg(self, subject=None):
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

        # run GLM on EPI prediction
        pred_epi_fitted = self.run_glm(
            avg_tc,
            self.sb["epi_pred_glm"],
            convolve_hrf=False
        )

        # get r2
        r2_epi = r2_score(avg_tc,pred_epi_fitted)
        # utils.verbose(f" r2 of GLM with on EPI prediction: {r2_epi}", self.verbose)
        for ii,par in zip(list(self.df_subj_r2.keys()),["epi",1,r2_epi]):
            self.df_subj_r2[ii].append(par)

        for ii,par in zip(list(self.df_subj_r2.keys()),["line",2,getattr(self.sb["line_obj"], f"{self.model}_iter")[0,-1]]):
            self.df_subj_r2[ii].append(par)                

        self.df_subj_r2 = pd.DataFrame(self.df_subj_r2)
        self.df_subj_r2["subject"] = subject

        # put preditions in dataframe
        self.df_subj_pred = {}
        for ii,par in zip(
            ["epi_pred_glm","line_pred","epi_block","epi_pred","tc_bold"],
            [pred_epi_fitted,self.sb["line_pred"],pred_block,self.sb["epi_pred"],self.sb["tc_bold"]]):
            self.df_subj_pred[ii] = np.squeeze(par)

        self.df_subj_pred = pd.DataFrame(self.df_subj_pred)
        self.df_subj_pred["subject"] = subject
        self.df_subj_pred["t"] = list(np.arange(0,self.df_subj_pred.iloc[:,0].values.shape[0])*self.TR)
        self.df_subj_pred = self.df_subj_pred.set_index(["subject","t"])

        # get predicted parameters
        self.df = []
        for ix,ii in enumerate(["line_pars","epi_pars"]):
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

    def plot_block_vs_epi(self, axs=None, exclude_line=True, posthoc=False, **kwargs):

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
            y_label2="variance explained (r2)",
            lim=[0,1],
            ticks=list(np.arange(0,1.2,0.2)),
            fancy=True,
            trim_bottom=True)

        # run posthoc?
        if posthoc:
            self.posth = Posthoc(
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
                    "width_ratios": [0.5,1,ww]
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

        # make prfs
        plotting.LazyPRF(
            np.zeros((500,500)), 
            cmap=utils.make_binary_cm(self.color[-1]),
            vf_extent=vf_extent,
            cross_color="k",
            edge_color=None,
            shrink_factor=0.9,
            vf_only=True,
            ax=ax1)

        for ii,col in zip([0,1],self.color[-2:]):

            pars = utils.select_from_df(self.pars_for_plot, expression=f"code = {ii}").values
            circ = plt.Circle(
                (pars[0,0],pars[0,1]),
                pars[0,2],
                ec=col,
                lw=2,
                fill=False)

            ax1.add_artist(circ)

        for ii,val in zip(["-5°","5°","-5°","5°"], [(0,0.51),(0.98,0.51),(0.51,0),(0.51,0.96)]):
            ax1.annotate(
                ii,
                val,
                fontsize=self.plot_defaults.label_size,
                xycoords="axes fraction"
            )            

        self.x_axis = np.array(list(np.arange(0,self.data_for_plot.shape[0])*self.TR))
        plotting.LazyPlot(
            [self.data_for_plot[qq].values for qq in ["tc_bold","line_pred","epi_pred_glm"]],
            xx=self.x_axis,
            add_hline="default",
            color=self.color,
            line_width=[1,3,3],
            markers=['.',None,None],
            x_label="time (s)",
            y_label="amplitude",
            labels=['data','line','epi'],
            axs=ax2,
            x_lim=[0,int(self.x_axis[-1])],
            x_ticks=list(np.arange(0,self.x_axis[-1]+40,40)))

        # bar plot with epi pRF vs design as block
        self.plot_block_vs_epi(axs=ax3, **kwargs)

        # move the timecourse plot towards middle
        plt.tight_layout()

        box = ax2.get_position()
        box.x0 -= 0.03
        box.x1 -= 0.03
        ax2.set_position(box)

        # add annotations
        plotting.fig_annot(
            self.fig,
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
        for ix,it in zip(["epi","line"],[0,2]):
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

class Posthoc(plotting.Defaults):

    def __init__(
        self,
        df=None,
        dv=None,
        between=None,
        parametric=True,
        padjust="fdr_bh",
        effsize="cohen",
        axs=None,
        alpha=0.05):

        super().__init__()

        self.df = df
        self.dv = dv
        self.between = between
        self.parametric = parametric
        self.padjust = padjust
        self.effsize = effsize
        self.axs = axs
        self.alpha = alpha

    def run_posthoc(self):

        try:
            import pingouin
        except:
            raise ImportError(f"Could not import 'pingouin'")

        # FDR-corrected post hocs with Cohen's D effect size
        self.posthoc = pingouin.pairwise_tests(
            data=self.df, 
            dv=self.dv, 
            between=self.between, 
            parametric=self.between, 
            padjust=self.padjust, 
            effsize=self.effsize)

    def plot_bars(self):
        
        if not hasattr(self, "posthoc"):
            self.run_posthoc()

        self.minmax = list(self.axs.get_ylim())
        self.y_pos = 0.95
        self.conditions = np.unique(self.df[self.between].values)

        # sort posthoc so that bars furthest away are on top (if significant)
        self.posthoc_sorted = sort_posthoc(self.posthoc)

        if "p-corr" in list(self.posthoc_sorted.columns):
            p_meth = "p-corr"
        else:
            p_meth = "p-unc"

        if not all((self.posthoc_sorted[p_meth]<self.alpha) == False):
            for contr in range(self.posthoc_sorted.shape[0]):
                if self.posthoc_sorted[p_meth].iloc[contr]<self.alpha:
                    
                    txt = "*"
                    style = None

                    # read indices from output dataframe and conditions
                    A = self.posthoc_sorted["A"].iloc[contr]
                    B = self.posthoc_sorted["B"].iloc[contr]

                    x1 = np.where(self.conditions == A)[0][0]
                    x2 = np.where(self.conditions == B)[0][0]

                    diff = self.minmax[1]-self.minmax[0]
                    m = self.minmax[1]
                    y, h, col =  (diff*self.y_pos)+self.minmax[0], diff*0.02, 'k'
                    self.axs.plot(
                        [x1,x1,x2,x2], 
                        [y,y+h,y+h,y], 
                        lw=self.tick_width, 
                        c=col)

                    self.axs.text(
                        (x1+x2)*.5, 
                        y+h*0.5, 
                        txt, 
                        ha='center', 
                        va='bottom', 
                        color=col,
                        fontsize=self.font_size,
                        style=style)

                    # make subsequent bar lower than first
                    self.y_pos -= 0.065
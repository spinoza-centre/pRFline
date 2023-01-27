from linescanning import(
    plotting,
    utils,
    prf
)
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
            epi_df = prf.SizeResponse.parameters_to_df(epi_avg, model=self.model)
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

                pars_df = prf.SizeResponse.parameters_to_df(pars, model=self.model)
                pars_df["subject"],pars_df["run"],pars_df["acq"],pars_df["code"] = sub, run+1,"epi runs",1

                self.df_params.append(pars_df)

            # do lines
            avg_line,_,_,_ = line_obj.plot_vox(
                model=line_obj.model,
                stage='iter',
                make_figure=False
            )

            line_df = prf.SizeResponse.parameters_to_df(avg_line, model=self.model)
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

                pars_df = prf.SizeResponse.parameters_to_df(pars, model=self.model)
                pars_df["subject"],pars_df["run"],pars_df["acq"],pars_df["code"] = sub, f"depth-{depth+1}","ribbon",3

                self.df_params.append(pars_df)

            # do full line
            pars = getattr(full_obj, f"{self.model}_iter").copy()
            vox_range = np.arange(pars.shape[0])
            pars_df = prf.SizeResponse.parameters_to_df(pars, model=self.model)
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

        # make annotations
        dist_from_x0 = 1/(self.fig.get_size_inches()*self.fig.dpi)[0]*50
        for ix,ax in enumerate(self.fig.axes):

            if ix < 1:
                pos = ax.get_position().x0-0.075
            else:
                pos = ax.get_position().x0-0.09
        
            ax.annotate(
                self.alphabet[ix], 
                (pos,1.01), 
                fontsize=28, 
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

        if not all((self.posthoc_sorted["p-corr"]<self.alpha) == False):
            for contr in range(self.posthoc_sorted.shape[0]):
                if self.posthoc_sorted["p-corr"].iloc[contr]<self.alpha:
                    
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
                    self.y_pos -= 0.05
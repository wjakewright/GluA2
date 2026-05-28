import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sy

from scipy import stats

from collections import defaultdict

from plotting_functions.plot_bar_plot import plot_bar_plot
from plotting_functions.plot_grouped_bar_plot import plot_grouped_bar_plot
from plotting_functions.plot_heatmap import plot_general_heatmap

sns.set_style("ticks")

# Set up area lists
macro_areas = {
    "Isocortex": "Isocortex",
    "TH": "Thalamus",
    "HPF": "Hippocampal formation",
    "MB": "Midbrain",
    "STR": "Striatum",
    "CTXsp": "Cortical subplate",
    "PAL": "Pallidum",
    "MY": "Medulla",
    "CB": "Cerebellum",
    "OLF": "Olfactory areas",
    "P": "Pons",
    "HY": "hypothalamus",
}

major_cortical_areas = {
    "FRP": "Frontal pole",
    "MOp": "Primary motor area",
    "MOs": "Secondary motor area",
    "SSp": "Primary somatosensory area",
    "SSs": "Supplemental somatosensory area",
    "GU": "Gustatory areas",
    "VISC": "Visceral area",
    "AUDd": "Dorsal auditory area",
    "AUDp": "Primary auditory area",
    "AUDpo": "Posterior auditory area",
    "AUDv": "Ventral auditory area",
    "VISal": "Anterolateral visual area",
    "VISam": "Anteromedial visual area",
    "VISl": "Lateral visual area",
    "VISp": "Primary visual area",
    "VISpl": "Posterolateral visual area",
    "VISpm": "Posteromedial visual area",
    "ACA": "Anterior cingulate area",
    "PL": "Prelimbic",
    "ILA": "Infralimbic",
    "ORB": "Orbital area",
    "AI": "Angular insular area",
    "RSP": "Retrosplenial area",
    "PTLp": "Posterior parietal association areas",
    "TEa": "Temporal association areas",
    "PERI": "Peririhnal area",
    "ECT": "Ectorihnal area",
}

major_olfactory_areas = {
    "MOB": "Main olfactory bulb",
    "AOB": "Accessory olfactory bulb",
    "AON": "Anterior olfactory nucleus",
    "TT": "Taenia tecta",
    "DP": "Dorsal peduncular area",
    "PIR": "Piriform area",
    "NLOT": "Nucleus of the lateral olfactory tract",
    "COA": "Cortical amygdalar area",
    "PAA": "Piriform amygdala area",
    "TR": "Postpirform transition area",
}

major_hippocampal_areas = {
    "CA_1": "Field CA1",
    "CA2": "Field CA2",
    "CA3": "Field CA3",
    "DG": "Dentate gyrus",
    "FC": "Fasciola cinerea",
    "ENT": "Entorhinal area",
    "PAR": "Parasubiculum",
    "POST": "Postsubiculum",
    "SUB": "Subiculum",
}

cortical_subplate_areas = {
    "CLA": "Claustrum",
    "EP": "Endopiriform nucleus",
    "LA": "Lateral amygdala nucleus",
    "BLA": "Basolateral amygdala nucleus",
    "BMA": "Basomedial amygdala nucleus",
    "PA": "Posterior amygdala nucleus",
}

major_forebrain_nuclei = {
    "STRd": "Dorsal striatum",
    "STRv": "Ventral striatum",
    "ACB": "Nucleus accumbens",
    "OT": "Olfactory tubercle",
    "LS": "Lateral septal nucleus",
    "SF": "Septofimbrial nucleus",
    "SH": "Septohippocampal nucleus",
    "AAA": "Anterior amygdala area",
    "BA": "Bed nucleus of the accessory olfactory tract",
    "CEA": "Central amygdala nucleus",
    "IA": "Intercalated amygdala nucleus",
    "MEA": "Medial amygdalar nucleus",
    "GPe": "External globus pallidum",
    "GPi": "Internal globus pallidum",
    "PALv": "Ventral pallidum",
    "PALm": "Meidal pallidum",
    "BST": "Bed nuclei of the stria terminalis",
}

major_thalamic_areas = {
    "VAL": "Ventral anterior-lateral complex",
    "VM": "Ventral medial nucleus",
    "VP": "Ventral posterior nucleus",
    "SPF": "Subparafascicular nucleus",
    "PP": "Peripeduncular nucleus",
    "MG": "Medial geniculate complex",
    "LGd": "Lateral geniculate complex",
    "LAT": "Lateral group nuclei",
    "ATN": "Anterior group nuclei",
    "MED": "Mediodorsal nuclei",
    "MTN": "Midline nuclei",
    "ILM": "Intralaminar nuclei",
    "RT": "Reticular nucleus",
    "GENv": "Ventral nuclei",
    "MH": "Medial habenula",
    "LH": "Lateral habenula",
}

major_hypothalamus_areas = {
    "PVH": "Paraventricular nucleus",
    "ARH": "Arcuate nucleus",
    "ADP": "Anterodorsal preoptic nucleus",
    "AVP": "Anteroventral preoptic nucleus",
    "AVPV": "Anteroventral periventricular nucleus",
    "DMH": "Dorsalmedial nucleus",
    "MEPO": "Median preoptic nucleus",
    "MPO": "Medial preoptic area",
    "PD": "Posterodorsal preoptic nucleus",
    "PS": "Parastrial nucleus",
    "PVp": "Posterior Paraventricular nucleus",
    "SBPV": "Subparaventricular zone",
    "SCH": "Suprachiasmatic nucleus",
    "SFO": "Subfornical organ",
    "VLPO": "Ventrolateral preoptic nucleus",
    "AHN": "Anterior hypothalamic nucleus",
    "MBO": "Mammilary body",
    "MPN": "Medial preoptic nucleus",
    "PVHd": "Paraventricular hypothalamic nucleus descending",
    "VMH": "Ventromedial hypothalamic nucleus",
    "LHA": "Lateral hypothalamic area",
    "LPO": "Lateral preoptic area",
    "PH": "Posterior hypothalamic nucleus",
    "STN": "Subthalamic nucleus",
    "ZI": "Zona incerta",
}

major_midbrain_areas = {
    "SCs": "Superior colliculus",
    "IC": "Inferior colliculus",
    "SNr": "Substantia nigra reticular",
    "VTA": "Ventral tegmental area",
    "MRN": "Midbrain reticular nucleus",
    "SCm": "Superior colliculus motor",
    "PAG": "Periaqueductal gray",
    "PRT": "Pretectal region",
    "CUN": "Cuneiform nucleus",
    "RN": "Red nucleus",
    "III": "Oculomotor nucleus",
    "EW": "Edinger-Westphal nucleus",
    "SNc": "Substantia nigra pars compacata",
    "PPN": "Pedunculopontine nucleus",
    "DR": "Dorsal raphe nucleus",
}


def compare_raw_lifetimes(
    untrained_data,
    trained_data,
    figsize=(15, 15),
    display_stats=False,
    save=False,
    save_path=None,
):
    """Figure to plot and compare the raw lifetime values across brain regions
    for trained and untrained mice

    INPUT PARAMETERS
        untrained_data - list of pd.dataframe containing the analyzed image data
                            for untrained mice

        trained_data - list of pd.dataframe containing the analyzed image data
                        for trained mice

        figsize - tuple sepcifying how large to make the figure

        display_stats - boolean specifying whether to perform stats

        save - boolean specifying whether to save the figure

        save_path - str specifying where to save the figure

    """
    COLORS = ["silver", "mediumseagreen"]

    # Grab the data
    trained_macro_areas = collect_data_values(
        trained_data, macro_areas, var_name="mean_lifetime"
    )
    untrained_macro_areas = collect_data_values(
        untrained_data, macro_areas, var_name="mean_lifetime"
    )
    trained_cortical_areas = collect_data_values(
        trained_data, major_cortical_areas, var_name="mean_lifetime"
    )
    untrained_cortical_areas = collect_data_values(
        untrained_data, major_cortical_areas, var_name="mean_lifetime"
    )
    trained_olfactory_areas = collect_data_values(
        trained_data, major_olfactory_areas, var_name="mean_lifetime"
    )
    untrained_olfactory_areas = collect_data_values(
        untrained_data, major_olfactory_areas, var_name="mean_lifetime"
    )
    trained_hippocampal_areas = collect_data_values(
        trained_data, major_hippocampal_areas, var_name="mean_lifetime"
    )
    untrained_hippocampal_areas = collect_data_values(
        untrained_data, major_hippocampal_areas, var_name="mean_lifetime"
    )
    trained_subplate_areas = collect_data_values(
        trained_data, cortical_subplate_areas, var_name="mean_lifetime"
    )
    untrained_subplate_areas = collect_data_values(
        untrained_data, cortical_subplate_areas, var_name="mean_lifetime"
    )
    trained_forebrain_areas = collect_data_values(
        trained_data, major_forebrain_nuclei, var_name="mean_lifetime"
    )
    untrained_forebrain_areas = collect_data_values(
        untrained_data, major_forebrain_nuclei, var_name="mean_lifetime"
    )
    trained_thalamic_areas = collect_data_values(
        trained_data, major_thalamic_areas, var_name="mean_lifetime"
    )
    untrained_thalamic_areas = collect_data_values(
        untrained_data, major_thalamic_areas, var_name="mean_lifetime"
    )
    trained_hypothalamus_areas = collect_data_values(
        trained_data, major_hypothalamus_areas, var_name="mean_lifetime"
    )
    untrained_hypothalamus_areas = collect_data_values(
        untrained_data, major_hypothalamus_areas, var_name="mean_lifetime"
    )
    trained_midbrain_areas = collect_data_values(
        trained_data, major_midbrain_areas, var_name="mean_lifetime"
    )
    untrained_midbrain_areas = collect_data_values(
        untrained_data, major_midbrain_areas, var_name="mean_lifetime"
    )

    # Construct the subplots
    fig, axes = plt.subplot_mosaic(
        """
        A
        B
        C
        D
        E
        F
        G
        H
        I
        """,
        figsize=figsize,
    )

    title = "Raw_GluA2_Lifetime"
    fig.suptitle(title)
    fig.subplots_adjust(wspace=0.5, hspace=1.0)

    # Plot data onto axes
    ## Macro Areas
    plot_grouped_bar_plot(
        data_dict={
            "Untrained": untrained_macro_areas,
            "Trained": trained_macro_areas,
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Macro areas",
        xtitle=None,
        ytitle="GluA2 lifetime tau (d)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        gap=0.2,
        b_linewidth=0,
        b_alpha=1,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["A"],
        save=False,
        save_path=None,
    )

    ## Cortical Areas
    plot_grouped_bar_plot(
        data_dict={
            "Untrained": untrained_cortical_areas,
            "Trained": trained_cortical_areas,
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Cortical areas",
        xtitle=None,
        ytitle="GluA2 lifetime tau (d)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        gap=0.2,
        b_linewidth=0,
        b_alpha=1,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["B"],
        save=False,
        save_path=None,
    )

    ## Hippocampal Areas
    plot_grouped_bar_plot(
        data_dict={
            "Untrained": untrained_hippocampal_areas,
            "Trained": trained_hippocampal_areas,
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Hippocampal areas",
        xtitle=None,
        ytitle="GluA2 lifetime tau (d)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        gap=0.2,
        b_linewidth=0,
        b_alpha=1,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["C"],
        save=False,
        save_path=None,
    )

    ## Cortical subplate Areas
    plot_grouped_bar_plot(
        data_dict={
            "Untrained": untrained_subplate_areas,
            "Trained": trained_subplate_areas,
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Cortical subplate areas",
        xtitle=None,
        ytitle="GluA2 lifetime tau (d)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        gap=0.2,
        b_linewidth=0,
        b_alpha=1,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["D"],
        save=False,
        save_path=None,
    )

    ## Forebrain Areas
    plot_grouped_bar_plot(
        data_dict={
            "Untrained": untrained_forebrain_areas,
            "Trained": trained_forebrain_areas,
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Forebrain nuclei",
        xtitle=None,
        ytitle="GluA2 lifetime tau (d)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        gap=0.2,
        b_linewidth=0,
        b_alpha=1,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["E"],
        save=False,
        save_path=None,
    )

    ## Olfactory Areas
    plot_grouped_bar_plot(
        data_dict={
            "Untrained": untrained_olfactory_areas,
            "Trained": trained_olfactory_areas,
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Olfactory areas",
        xtitle=None,
        ytitle="GluA2 lifetime tau (d)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        gap=0.2,
        b_linewidth=0,
        b_alpha=1,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["F"],
        save=False,
        save_path=None,
    )

    ## Thalamic Areas
    plot_grouped_bar_plot(
        data_dict={
            "Untrained": untrained_thalamic_areas,
            "Trained": trained_thalamic_areas,
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Thalamic areas",
        xtitle=None,
        ytitle="GluA2 lifetime tau (d)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        gap=0.2,
        b_linewidth=0,
        b_alpha=1,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["G"],
        save=False,
        save_path=None,
    )

    ## Hypothalamus Areas
    plot_grouped_bar_plot(
        data_dict={
            "Untrained": untrained_hypothalamus_areas,
            "Trained": trained_hypothalamus_areas,
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Hypothalamic areas",
        xtitle=None,
        ytitle="GluA2 lifetime tau (d)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        gap=0.2,
        b_linewidth=0,
        b_alpha=1,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["H"],
        save=False,
        save_path=None,
    )

    ## Midbrain Areas
    plot_grouped_bar_plot(
        data_dict={
            "Untrained": untrained_midbrain_areas,
            "Trained": trained_midbrain_areas,
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Midbrain areas",
        xtitle=None,
        ytitle="GluA2 lifetime tau (d)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        gap=0.2,
        b_linewidth=0,
        b_alpha=1,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["I"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")
        fig.savefig(fname + ".svg")

    # Statistics section
    if not display_stats:
        return


def compare_relative_lifetimes(
    untrained_data,
    trained_data,
    figsize=(15, 15),
    display_stats=False,
    save=False,
    save_path=None,
):
    """
    Figure to plot and compare the raw lifetime values across brain regions
    for trained and untrained mice

    INPUT PARAMETERS
        untrained_data - list of pd.dataframe containing the analyzed image data
                            for untrained mice

        trained_data - list of pd.dataframe containing the analyzed image data
                        for trained mice

        figsize - tuple sepcifying how large to make the figure

        display_stats - boolean specifying whether to perform stats

        save - boolean specifying whether to save the figure

        save_path - str specifying where to save the figure
    """

    color = ["mediumseagreen"]

    # Grab the data
    trained_macro_areas = collect_data_values(
        trained_data, macro_areas, var_name="mean_lifetime"
    )
    untrained_macro_areas = collect_data_values(
        untrained_data, macro_areas, var_name="mean_lifetime"
    )
    trained_cortical_areas = collect_data_values(
        trained_data, major_cortical_areas, var_name="mean_lifetime"
    )
    untrained_cortical_areas = collect_data_values(
        untrained_data, major_cortical_areas, var_name="mean_lifetime"
    )
    trained_olfactory_areas = collect_data_values(
        trained_data, major_olfactory_areas, var_name="mean_lifetime"
    )
    untrained_olfactory_areas = collect_data_values(
        untrained_data, major_olfactory_areas, var_name="mean_lifetime"
    )
    trained_hippocampal_areas = collect_data_values(
        trained_data, major_hippocampal_areas, var_name="mean_lifetime"
    )
    untrained_hippocampal_areas = collect_data_values(
        untrained_data, major_hippocampal_areas, var_name="mean_lifetime"
    )
    trained_subplate_areas = collect_data_values(
        trained_data, cortical_subplate_areas, var_name="mean_lifetime"
    )
    untrained_subplate_areas = collect_data_values(
        untrained_data, cortical_subplate_areas, var_name="mean_lifetime"
    )
    trained_forebrain_areas = collect_data_values(
        trained_data, major_forebrain_nuclei, var_name="mean_lifetime"
    )
    untrained_forebrain_areas = collect_data_values(
        untrained_data, major_forebrain_nuclei, var_name="mean_lifetime"
    )
    trained_thalamic_areas = collect_data_values(
        trained_data, major_thalamic_areas, var_name="mean_lifetime"
    )
    untrained_thalamic_areas = collect_data_values(
        untrained_data, major_thalamic_areas, var_name="mean_lifetime"
    )
    trained_hypothalamus_areas = collect_data_values(
        trained_data, major_hypothalamus_areas, var_name="mean_lifetime"
    )
    untrained_hypothalamus_areas = collect_data_values(
        untrained_data, major_hypothalamus_areas, var_name="mean_lifetime"
    )
    trained_midbrain_areas = collect_data_values(
        trained_data, major_midbrain_areas, var_name="mean_lifetime"
    )
    untrained_midbrain_areas = collect_data_values(
        untrained_data, major_midbrain_areas, var_name="mean_lifetime"
    )

    macro_relative = calculate_relative_values(
        untrained_macro_areas, trained_macro_areas
    )
    cortical_relative = calculate_relative_values(
        untrained_cortical_areas, trained_cortical_areas
    )
    hippocampal_relative = calculate_relative_values(
        untrained_hippocampal_areas, trained_hippocampal_areas
    )
    subplate_relative = calculate_relative_values(
        untrained_subplate_areas, trained_subplate_areas
    )
    forebrain_relative = calculate_relative_values(
        untrained_forebrain_areas, trained_forebrain_areas
    )
    olfactory_relative = calculate_relative_values(
        untrained_olfactory_areas, trained_olfactory_areas
    )
    thalamic_relative = calculate_relative_values(
        untrained_thalamic_areas, trained_thalamic_areas
    )
    hypothalamic_relative = calculate_relative_values(
        untrained_hypothalamus_areas, trained_hypothalamus_areas
    )
    midbrain_relative = calculate_relative_values(
        untrained_midbrain_areas, trained_midbrain_areas
    )

    # Construct the subplots
    fig, axes = plt.subplot_mosaic(
        """
        A
        B
        C
        D
        E
        F
        G
        H
        I
        """,
        figsize=figsize,
    )

    title = "Relative_GluA2_Lifetime"
    fig.suptitle(title)
    fig.subplots_adjust(wspace=0.5, hspace=1.0)

    plot_bar_plot(
        data_dict=macro_relative,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Major areas",
        xtitle=None,
        ytitle="Relative GluA2 lifetime (%)",
        ylim=None,
        b_colors=color,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1,
        b_alpha=1,
        s_colors=color,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["A"],
        save=False,
        save_path=None,
    )

    plot_bar_plot(
        data_dict=cortical_relative,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Cortical areas",
        xtitle=None,
        ytitle="Relative GluA2 lifetime (%)",
        ylim=None,
        b_colors=color,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1,
        b_alpha=1,
        s_colors=color,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["B"],
        save=False,
        save_path=None,
    )

    plot_bar_plot(
        data_dict=hippocampal_relative,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Hippocampal areas",
        xtitle=None,
        ytitle="Relative GluA2 lifetime (%)",
        ylim=None,
        b_colors=color,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1,
        b_alpha=1,
        s_colors=color,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["C"],
        save=False,
        save_path=None,
    )

    plot_bar_plot(
        data_dict=subplate_relative,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Cortical subplate areas",
        xtitle=None,
        ytitle="Relative GluA2 lifetime (%)",
        ylim=None,
        b_colors=color,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1,
        b_alpha=1,
        s_colors=color,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["D"],
        save=False,
        save_path=None,
    )

    plot_bar_plot(
        data_dict=forebrain_relative,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Forebrain nuclei",
        xtitle=None,
        ytitle="Relative GluA2 lifetime (%)",
        ylim=None,
        b_colors=color,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1,
        b_alpha=1,
        s_colors=color,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["E"],
        save=False,
        save_path=None,
    )

    plot_bar_plot(
        data_dict=olfactory_relative,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Olfactory areas",
        xtitle=None,
        ytitle="Relative GluA2 lifetime (%)",
        ylim=None,
        b_colors=color,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1,
        b_alpha=1,
        s_colors=color,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["F"],
        save=False,
        save_path=None,
    )

    plot_bar_plot(
        data_dict=thalamic_relative,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Thalamic areas",
        xtitle=None,
        ytitle="Relative GluA2 lifetime (%)",
        ylim=None,
        b_colors=color,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1,
        b_alpha=1,
        s_colors=color,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["G"],
        save=False,
        save_path=None,
    )

    plot_bar_plot(
        data_dict=hypothalamic_relative,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Hypothalamic areas",
        xtitle=None,
        ytitle="Relative GluA2 lifetime (%)",
        ylim=None,
        b_colors=color,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1,
        b_alpha=1,
        s_colors=color,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    plot_bar_plot(
        data_dict=midbrain_relative,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Midbrain areas",
        xtitle=None,
        ytitle="Relative GluA2 lifetime (%)",
        ylim=None,
        b_colors=color,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1,
        b_alpha=1,
        s_colors=color,
        s_size=5,
        s_alpha=1,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["I"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")
        fig.savefig(fname + ".svg")

    # Statistics section
    if not display_stats:
        return


def plot_interregion_correlations(
    untrained_data,
    trained_data,
    figsize=(15, 15),
    display_stats=False,
    save=False,
    save_path=None,
):
    """
    Figure to correlate and plot changes in lifetimes across brain regions

    INPUT PARAMETERS
        untrained_data - list of pd.dataframe containing the analyzed image data
                            for untrained mice

        trained_data - list of pd.dataframe containing the analyzed image data
                        for trained mice

        figsize - tuple sepcifying how large to make the figure

        display_stats - boolean specifying whether to perform stats

        save - boolean specifying whether to save the figure

        save_path - str specifying where to save the figure

    """

    trained_cortical_areas = collect_data_values(
        trained_data, major_cortical_areas, var_name="mean_lifetime"
    )
    untrained_cortical_areas = collect_data_values(
        untrained_data, major_cortical_areas, var_name="mean_lifetime"
    )
    trained_olfactory_areas = collect_data_values(
        trained_data, major_olfactory_areas, var_name="mean_lifetime"
    )
    untrained_olfactory_areas = collect_data_values(
        untrained_data, major_olfactory_areas, var_name="mean_lifetime"
    )
    trained_hippocampal_areas = collect_data_values(
        trained_data, major_hippocampal_areas, var_name="mean_lifetime"
    )
    untrained_hippocampal_areas = collect_data_values(
        untrained_data, major_hippocampal_areas, var_name="mean_lifetime"
    )
    trained_subplate_areas = collect_data_values(
        trained_data, cortical_subplate_areas, var_name="mean_lifetime"
    )
    untrained_subplate_areas = collect_data_values(
        untrained_data, cortical_subplate_areas, var_name="mean_lifetime"
    )
    trained_forebrain_areas = collect_data_values(
        trained_data, major_forebrain_nuclei, var_name="mean_lifetime"
    )
    untrained_forebrain_areas = collect_data_values(
        untrained_data, major_forebrain_nuclei, var_name="mean_lifetime"
    )
    trained_thalamic_areas = collect_data_values(
        trained_data, major_thalamic_areas, var_name="mean_lifetime"
    )
    untrained_thalamic_areas = collect_data_values(
        untrained_data, major_thalamic_areas, var_name="mean_lifetime"
    )
    trained_hypothalamus_areas = collect_data_values(
        trained_data, major_hypothalamus_areas, var_name="mean_lifetime"
    )
    untrained_hypothalamus_areas = collect_data_values(
        untrained_data, major_hypothalamus_areas, var_name="mean_lifetime"
    )
    trained_midbrain_areas = collect_data_values(
        trained_data, major_midbrain_areas, var_name="mean_lifetime"
    )
    untrained_midbrain_areas = collect_data_values(
        untrained_data, major_midbrain_areas, var_name="mean_lifetime"
    )

    cortical_relative = calculate_relative_values(
        untrained_cortical_areas, trained_cortical_areas
    )
    hippocampal_relative = calculate_relative_values(
        untrained_hippocampal_areas, trained_hippocampal_areas
    )
    subplate_relative = calculate_relative_values(
        untrained_subplate_areas, trained_subplate_areas
    )
    forebrain_relative = calculate_relative_values(
        untrained_forebrain_areas, trained_forebrain_areas
    )
    olfactory_relative = calculate_relative_values(
        untrained_olfactory_areas, trained_olfactory_areas
    )
    thalamic_relative = calculate_relative_values(
        untrained_thalamic_areas, trained_thalamic_areas
    )
    hypothalamic_relative = calculate_relative_values(
        untrained_hypothalamus_areas, trained_hypothalamus_areas
    )
    midbrain_relative = calculate_relative_values(
        untrained_midbrain_areas, trained_midbrain_areas
    )

    # Join the dictionaries
    all_relative = {
        **cortical_relative,
        **hippocampal_relative,
        **subplate_relative,
        **forebrain_relative,
        **olfactory_relative,
        **thalamic_relative,
        **hypothalamic_relative,
        **midbrain_relative,
    }

    all_trained_areas = {
        **trained_cortical_areas,
        **trained_hippocampal_areas,
        **trained_subplate_areas,
        **trained_forebrain_areas,
        **trained_olfactory_areas,
        **trained_thalamic_areas,
        **trained_hypothalamus_areas,
        **trained_midbrain_areas,
    }

    all_untrained_areas = {
        **untrained_cortical_areas,
        **untrained_hippocampal_areas,
        **untrained_subplate_areas,
        **untrained_forebrain_areas,
        **untrained_olfactory_areas,
        **untrained_thalamic_areas,
        **untrained_hypothalamus_areas,
        **untrained_midbrain_areas,
    }

    # Perform the correlations
    trained_correlations = correlate_relative_changes(all_trained_areas)

    trained_sorted_correlations = cluster_array(trained_correlations)

    untrained_correlations = correlate_relative_changes(all_untrained_areas)

    untrained_sorted_correlations = cluster_array(trained_correlations)

    relative_correlations = correlate_relative_changes(all_relative)

    relative_sorted_correlations = cluster_array(relative_correlations)

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABC
        """,
        figsize=figsize,
    )

    title = "GluA2_Lifetime_Correlations"
    fig.suptitle(title)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    # Plot the data
    axes["A"].set_aspect(aspect="equal", adjustable="box")
    plot_general_heatmap(
        data=untrained_sorted_correlations,
        figsize=(5, 5),
        title="Untrained",
        xtitle=None,
        ytitle=None,
        cbar_label="Lifetime change",
        hmap_range=(0, 1),
        center=None,
        cmap="plasma",
        axis_width=2.5,
        minor_ticks=None,
        tick_len=3,
        annotate=False,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    axes["B"].set_aspect(aspect="equal", adjustable="box")
    plot_general_heatmap(
        data=trained_sorted_correlations,
        figsize=(5, 5),
        title="Trained",
        xtitle=None,
        ytitle=None,
        cbar_label="Lifetime change",
        hmap_range=(0, 1),
        center=None,
        cmap="plasma",
        axis_width=2.5,
        minor_ticks=None,
        tick_len=3,
        annotate=False,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    axes["C"].set_aspect(aspect="equal", adjustable="box")
    plot_general_heatmap(
        data=relative_sorted_correlations,
        figsize=(5, 5),
        title="Relative",
        xtitle=None,
        ytitle=None,
        cbar_label="Lifetime change",
        hmap_range=(0, 1),
        center=None,
        cmap="bwr",
        axis_width=2.5,
        minor_ticks=None,
        tick_len=3,
        annotate=False,
        ax=axes["C"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")
        fig.savefig(fname + ".svg")

    # Statistics section
    if not display_stats:
        return


def collect_data_values(data_list, roi_list, var_name):
    """Helper function to grab and organize the data"""
    # Set up dictionary to collect the data
    organized_data = defaultdict(list)
    # Iterate through each mouse's data
    for data in data_list:
        # iterate through each brain region
        for key, value in roi_list.items():
            # Grabe the data
            roi_data = data.loc[data["roi_name"] == key, var_name]
            ## Average across the two hemispheres
            roi_data = np.nanmean(roi_data)
            ## Store the data
            organized_data[value].append(roi_data)

    return organized_data


def calculate_relative_values(control_dict, exp_dict):
    """Helper function to calculate relative change for each brain area"""
    # Set up the output
    relative_data = {}

    # Iterate through each roi
    for key, value in control_dict.items():
        ## Get the corresponding data from the experimental data
        exp_value = exp_dict[key]
        # Average the control values
        ctl_avg = np.nanmean(value)
        if np.isnan(ctl_avg):
            ctl_avg = 0
        # Calculate the percent difference
        relative_diff = (ctl_avg / np.array(exp_value)) * 100
        relative_diff = relative_diff - 100
        # Store the data
        relative_data[key] = relative_diff

    return relative_data


def correlate_relative_changes(relative_dict):
    """Helper function to correlate changes across brain regions"""

    names = list(relative_dict.keys())

    correlations = pd.DataFrame(columns=names, index=names)

    for region1 in names:
        for region2 in names:
            value1 = np.array(relative_dict[region1])
            value2 = np.array(relative_dict[region2])
            if np.isnan(value1).all():
                r = np.nan
            elif np.isnan(value2).all():
                r = np.nan
            else:
                value1[np.isnan(value1)] = 0.0
                value2[np.isnan(value2)] = 0.0

                r, _ = stats.spearmanr(value1, value2)
            correlations.at[region1, region2] = r

    return correlations


def cluster_array(array):
    """Helper function to cluster array data using hierarchical clustering"""

    np_array = array.to_numpy().astype(float)

    distances = sy.cluster.hierarchy.distance.pdist(np_array)
    try:
        linkage = sy.cluster.hierarchy.linkage(distances, method="complete")
    except:
        print(np_array)
        print("")
        print(distances)
        raise
    ind = sy.cluster.hierarchy.fcluster(linkage, 0.1 * distances.max(), "distance")

    ordered_ind = np.argsort(ind)
    sorted_array = pd.DataFrame(np_array)
    sorted_array = sorted_array.iloc[ordered_ind, ordered_ind]

    return sorted_array

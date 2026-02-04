import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

from plotting_functions.plot_bar_plot import plot_bar_plot
from plotting_functions.plot_grouped_bar_plot import plot_grouped_bar_plot

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
    "IL": "Infralimbic",
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
    "CA1": "Field CA1",
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
    COLORS = ["lightgreen", "darkgreen"]

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
        b_width=0.6,
        gap=0.1,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
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
        b_width=0.6,
        gap=0.1,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
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
        b_width=0.6,
        gap=0.1,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
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
        b_width=0.6,
        gap=0.1,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
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
        b_width=0.6,
        gap=0.1,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
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
        b_width=0.6,
        gap=0.1,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
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
        b_width=0.6,
        gap=0.1,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
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
        b_width=0.6,
        gap=0.1,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
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
        b_width=0.6,
        gap=0.1,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
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
        # Calculate the percent difference
        relative_diff = (np.array(exp_value) / ctl_avg) * 100
        # Store the data
        relative_data[key] = relative_diff

    return relative_data

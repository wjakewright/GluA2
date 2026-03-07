import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import rgb2hex
from scipy import stats

from plotting_functions.adjust_axes import adjust_axes, get_axis_limit

sns.set_style("ticks")


def plot_grouped_bar_plot(
    data_dict,
    mean_type="mean",
    err_type="sem",
    figsize=(5, 5),
    title=None,
    xtitle=None,
    ytitle=None,
    ylim=None,
    b_colors="mediumblue",
    b_edgecolors="black",
    b_err_colors="black",
    b_width=0.5,
    gap=0.2,
    b_linewidth=0,
    b_alpha=1,
    s_colors="mediumblue",
    s_size=5,
    s_alpha=1,
    plot_ind=False,
    axis_width=1.5,
    minor_ticks=None,
    tick_len=3,
    x_rotation=45,
    ax=None,
    save=False,
    save_path=None,
):
    """General function for plotting a scatter bar plots that are grouped

    INPUT PARAMETERS
        data_dict - nested dictionary of data to be plotted. Outer keys are the groups
                    and inner keys are the subgroups (will be used as x values)

        mean_type - str specifying what central point to plot. Accepts, mean and median

        err_type - str specifying what type of error bars to plot. Accepts sem, std, and CI

        figsize - tuple specifying the size of the figure. Only used for independent figures

        title - str specifying the title of the plot

        xtitle - str specifying the title of the x axis

        ytitle - str specifying the title of the y axis

        ylim - tuple specifying the limits of the y axis

        b_colors - str or list of str specifying the colors of the main groups

        b_edgecolors - str specifying the color of the edges of the boxes

        b_err_colors - str specifying the color of the error bars

        b_width - float specifying the width of the individual bars

        b_gap - float specifying the gap between bars between groups

        b_linewidth - float specifying the bar plot edge widths

        b_alpha - float specifying the alpha of the bars

        s_colors - str or list of str specifying the color of the scatter points

        s_size - int specifying the size of the scatter points

        s_alpha - float specifying the alpha of the scatter points

        plot_ind - boolean of whether or not to plot the individual data points

        axis_width - int or float specifying how thick the axis lines should be

        minor_ticks - str specifying if minor ticks should be add to the x and/or y
                    axes. Takes "both", "x", and "y" as inputs.

        tick_len - int or float specifying how long the tick marks should be

        ax - axis object you wish the data to be plotted on. Useful for subplotting

        save - boolean specifying if you wish to save the figure or not

        save_path - str specifying the path of where to save the figure


    """
    # make list of colors if only one is provided
    if type(b_colors) == str:
        b_colors = [b_colors for i in range(len(list(data_dict.keys())))]
    if type(s_colors) == str:
        s_colors = [s_colors for i in range(len(list(data_dict.keys())))]

    # Check if axis was provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()
    else:
        save = False  # Don't wish to save if part of another plot

    if mean_type == "mean":
        mean_function = np.nanmean
    elif mean_type == "median":
        mean_function = np.nanmedian

    if err_type == "sem":
        err_function = lambda x: sem_error(x, mean_function)
    elif err_type == "std":
        err_function = lambda x: std_error(x, mean_function)
    elif err_type == "CI":
        err_function = lambda x: CI_error(x, mean_function)

    # Organize the data for plotting
    dfs = []
    for key, value in data_dict.items():
        df = pd.DataFrame(value)

        df = df.melt(var_name="X", value_name="Y")
        df["Group"] = [key for _ in range(len(df))]
        dfs.append(df)

    # Join the dataframes
    plot_df = pd.concat(dfs, ignore_index=True)

    # Plot the points if specified
    if plot_ind == True:
        sns.stripplot(
            x="X",
            y="Y",
            hue="Group",
            data=plot_df,
            palette=s_colors,
            alpha=s_alpha,
            size=s_size,
            dodge=True,
            ax=ax,
            clip_on=False,
            legend=False,
            zorder=2,
        )

    # Plot the bars plots now
    sns.barplot(
        x="X",
        y="Y",
        hue="Group",
        data=plot_df,
        palette=b_colors,
        estimator=mean_function,
        errorbar=err_function,
        dodge=True,
        width=b_width,
        gap=gap,
        alpha=b_alpha,
        err_kws={"color": b_err_colors},
        edgecolor=b_edgecolors,
        linewidth=b_linewidth,
        ax=ax,
        legend=True,
        zorder=1,
    )

    ax.set_title(title)

    # Format the axes
    adjust_axes(
        ax,
        minor_ticks,
        xtitle,
        ytitle,
        tick_len,
        axis_width,
    )
    for tick in ax.get_xticklabels():
        tick.set_rotation(x_rotation)

    ax.set_xmargin(0.01)
    ticks = ax.get_yticks()
    bottom, top = get_axis_limit(ylim, ticks)
    ax.set_ylim(bottom=bottom, top=top)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")


######## HEPLER FUNCTIONS ##########


def std_error(values, central_fun):
    center = central_fun(values)
    std = np.nanstd(values)
    upper = center + std
    lower = center - std

    return (upper, lower)


def sem_error(values, central_fun):
    center = central_fun(values)
    std = stats.sem(values, nan_policy="omit")
    upper = center + std
    lower = center - std

    return (upper, lower)


def CI_error(values, central_fun):
    data = (values,)
    central = central_fun(values)
    bootstrap = stats.bootstrap(
        data,
        central_fun,
        confidence_level=0.95,
        method="percentile",
        n_resamples=1000,
    )
    upper = bootstrap.confidence_interval.high - central
    lower = bootstrap.confidence_interval.low

    return (upper, lower)

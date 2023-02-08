"""
Created on Fri Oct 16 17:07:19 2020.

@author: hashemghanem
"""
import torch
import matplotlib as mpl
import matplotlib.pylab as pylab
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick


def plot(*args, xlabel=None, ylabel=None, legend_labels=None,
         title=None, line_width=2, marker_size=6, file_name=None):
    """Plot with linear scales on both x,y."""
    # the following 2 lines will be used in case of multy-curve figures
    # colors = ['y', 'b', 'g', 'r', 'brown', 'k', 'violet']

    # Colorblind friendly color cycle (found on git, test it to trust it)
    # cb_clr_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
    #                 '#f781bf', '#a65628', '#984ea3',
    #                 '#999999', '#e41a1c', '#dede00']

    # Colorblind friendly color cycles
    ccycles = ['tableau-colorblind10', 'seaborn-colorblind']
    # Colorblind friendly colormaps
    cmaps = ['gray', 'cividis', 'plasma', 'virdis', 'magma', 'inferno']
    # you can also get a colorblind friendly cycle by linearly sampling
    # a cmap which is cb friendly too. search for "use matplotlib color
    # map for color cycle" on stackoverflow.

    # markers in case needed
    mrk = ['s', 'v', 'o', 'x', '3', 'p', '|']

    # making the labels/ticks' font larger
    params = {'legend.fontsize': 'large',
              'axes.labelsize': 'xx-large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large'}
    pylab.rcParams.update(params)

    # Set the color style (cycle + map) you want. For some reason
    # It doesn't take the names in cmaps array above, sch fr plt styl sheet.
    # If you want to change just the color cycle, then search for how
    # to extract the colory cycle of a pyplot style and then use
    # rcparams['axes.prop_cycle']=extracted_cycle
    plt.style.use(ccycles[0])

    # Check if both x,y are passed in args, or just y.
    # First get y.
    if len(args) == 1:
        y = torch.tensor(args[0])
    elif len(args) == 2:
        y = torch.tensor(args[1])
    if len(y.shape) == 1:
        y = y[:, None]
    # Second, get x.
    if len(args) == 1:
        x = torch.arange(y.shape[0])[:, None].repeat(1, y.shape[1])
    elif len(args) == 2:
        x = torch.tensor(args[0])
    x.reshape(*y.shape)

    # Now, create the fig on which you will plot
    fig, ax = plt.subplots(constrained_layout=True)

    # if you are plotting the percent-accuracy
    # fmt = '%.0f%%'
    # yticks = mtick.FormatStrFormatter(fmt)
    # ax.yaxis.set_major_formatter(yticks)

    # start plotting
    for i in range(y.shape[1]):
        plt.plot(x[:, i], y[:, i], linewidth=line_width, marker=mrk[i],
                 markersize=marker_size, markerfacecolor="None")
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if legend_labels is not None:
        ax.legend(legend_labels)
    ax.grid()

    # in case there is a problem with the legends
    # handles, labels = ax.get_legend_handles_labels()
    # ind_perm = [0, 2, 1, 3]
    # handles = [handles[i] for i in ind_perm]
    # labels = [labels[i] for i in ind_perm]
    # ax.legend(handles, labels)

    # plt.xscale('log')

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()


def plotlogxy(*args, xlabel=None, ylabel=None, legend_labels=None,
              title=None, file_name=None):
    """Plot with log scale on both x,y."""

    # Colorblind friendly color cycles
    ccycles = ['tableau-colorblind10', 'seaborn-colorblind']

    # markers in case needed
    mrk = ['s', 'v', 'o', 'x', '3', 'p', '|']

    # making the labels/ticks' font larger
    params = {'legend.fontsize': 'large',
              'axes.labelsize': 'xx-large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large'}
    pylab.rcParams.update(params)

    # Set the color style (cycle + map) you want.
    plt.style.use(ccycles[0])

    # Check if both x,y are passed in args, or just y.
    # First get y.
    if len(args) == 1:
        y = torch.tensor(args[0])
    elif len(args) == 2:
        y = torch.tensor(args[1])
    if len(y.shape) == 1:
        y = y[:, None]
    # Second, get x.
    if len(args) == 1:
        x = torch.arange(y.shape[0])[:, None].repeat(1, y.shape[1])
    elif len(args) == 2:
        x = torch.tensor(args[0])
    x.reshape(*y.shape)

    # Now, create the fig on which you will plot
    fig, ax = plt.subplots(constrained_layout=True)

    linewidth = 2
    # start plotting
    for i in range(y.shape[1]):
        plt.plot(x[:, i], y[:, i], linewidth=linewidth, marker=mrk[i],
                 markersize=10, markerfacecolor="None")
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if legend_labels is not None:
        ax.legend(legend_labels)
    ax.grid()
    plt.xscale('log')
    plt.yscale('log')

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()


def plotlogy(*args, xlabel=None, ylabel=None, legend_labels=None,
             title=None, line_width=2, marker_size=10, file_name=None):
    """Plot with log scale on both x,y."""

    # Colorblind friendly color cycles
    ccycles = ['tableau-colorblind10', 'seaborn-colorblind']

    # markers in case needed
    mrk = ['s', 'v', 'o', 'x', '3', 'p', '|']

    # making the labels/ticks' font larger
    params = {'legend.fontsize': 'large',
              'axes.labelsize': 'xx-large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large'}
    pylab.rcParams.update(params)

    # Set the color style (cycle + map) you want.
    plt.style.use(ccycles[0])

    # Check if both x,y are passed in args, or just y.
    # First get y.
    if len(args) == 1:
        y = torch.tensor(args[0])
    elif len(args) == 2:
        y = torch.tensor(args[1])
    if len(y.shape) == 1:
        y = y[:, None]
    # Second, get x.
    if len(args) == 1:
        x = torch.arange(y.shape[0])[:, None].repeat(1, y.shape[1])
    elif len(args) == 2:
        x = torch.tensor(args[0])
    x.reshape(*y.shape)

    # Now, create the fig on which you will plot
    fig, ax = plt.subplots(constrained_layout=True)

    # start plotting
    for i in range(y.shape[1]):
        plt.plot(x[:, i], y[:, i], linewidth=line_width, marker=mrk[i],
                 markersize=marker_size, markerfacecolor="None")
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if legend_labels is not None:
        ax.legend(legend_labels)
    ax.grid()
    plt.yscale('log')

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()


def imshow(img, colorbar=True, xlabel=None, ylabel=None,
           title=None, file_name=None):
    """Imshow the matrix passed in img as an image."""
    # Colorblind friendly colormaps
    cmaps = ['gray', 'cividis', 'plasma', 'virdis', 'magma', 'inferno']

    # making the labels/ticks' font larger
    params = {'legend.fontsize': 'large',
              'axes.labelsize': 'xx-large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large'}
    pylab.rcParams.update(params)
    plt.imshow(img, cmap=cmaps[0])
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if colorbar is True:
        plt.colorbar()

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()

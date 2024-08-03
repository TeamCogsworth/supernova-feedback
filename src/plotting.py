import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import gala.dynamics as gd

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)


bottom_ylabel = {
    "distance": r"Fraction of SNe > $d$",
    "time": r"Fraction of SNe after $t$",
    "ejecta_mass": r"Fraction of ejecta mass > $d$"
}
    
xlabel = {
    "distance": "SN distance from parent cluster [pc]",
    "time": r"Time since cluster birth, $t \, [\rm Myr]$",
    "ejecta_mass": "SN distance from parent cluster [pc]",
}

top_ylabel = {
    "distance": "Number of SN",
    "time": r"SNe Rate $\rm [yr^{-1}]$",
    "ejecta_mass": r"Ejecta mass $[\rm M_\odot]$"
}

labels = ["Effectively Single", "Primary", "Secondary", "Merger Product"]
colours = ["skyblue", plt.cm.viridis(0.4), plt.cm.viridis(0.7), "gold"]

labels = list(reversed(labels))
colours = list(reversed(colours))

def weighted_median(values, weights):
    """https://stackoverflow.com/questions/20601872/numpy-or-scipy-to-calculate-weighted-median"""
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, 0.5 * c[-1])]]

def get_centres_widths(bins):
    bin_centres = np.array([(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)])
    widths = np.insert(bin_centres[1:] - bin_centres[:-1], -1, bin_centres[1] - bin_centres[0])
    return bin_centres, widths

def remove_white_stripes(patches):
    for patch in patches:
        for bar in patch:
            bar.set_edgecolor(bar.get_facecolor())
            
def get_data_and_weights(p, var, widths=None):
    # first the data
    if var in ["distance", "ejecta_mass"]:
        data = [np.concatenate((p.primary_sn_distances.to(u.pc).value[p.sn_1_singles],
                                p.secondary_sn_distances.to(u.pc).value[p.sn_2_singles])),
                 p.primary_sn_distances.to(u.pc).value[p.sn_1],
                 p.secondary_sn_distances.to(u.pc).value[p.sn_2],
                 np.concatenate((p.primary_sn_distances.to(u.pc).value[p.sn_1_merger],
                                 p.secondary_sn_distances.to(u.pc).value[p.sn_2_merger]))]
    elif var == "time":
        sn_time_1 = p.bpp["tphys"][(p.bpp["evol_type"] == 15) & (~p.bpp["bin_num"].isin(p.duplicate_sn))]
        sn_time_2 = p.bpp["tphys"][(p.bpp["evol_type"] == 16) & (~p.bpp["bin_num"].isin(p.duplicate_sn))]
        data = [np.concatenate((sn_time_1[p.sn_1_singles], sn_time_2[p.sn_2_singles])),
                sn_time_1[p.sn_1].values,
                sn_time_2[p.sn_2].values,
                np.concatenate((sn_time_1[p.sn_1_merger], sn_time_2[p.sn_2_merger]))]
    
    # then the weights
    if var == "ejecta_mass":
        ejecta_mass_1 = p.bpp["mass_1"].diff(-1).fillna(0.0)[(p.bpp["evol_type"] == 15) & (~p.bpp["bin_num"].isin(p.duplicate_sn))]
        ejecta_mass_2 = p.bpp["mass_2"].diff(-1).fillna(0.0)[(p.bpp["evol_type"] == 16) & (~p.bpp["bin_num"].isin(p.duplicate_sn))]
        weights = [np.concatenate((ejecta_mass_1[p.sn_1_singles],
                                   ejecta_mass_2[p.sn_2_singles])),
                   ejecta_mass_1[p.sn_1].values,
                   ejecta_mass_2[p.sn_2].values,
                   np.concatenate((ejecta_mass_1[p.sn_1_merger],
                                   ejecta_mass_2[p.sn_2_merger]))]
    elif var == "time":
        weights = [np.ones_like(d) / (widths[0] * 1e6) for d in data]
    else:
        weights = [np.ones_like(d) for d in data]
        
        
    data = list(reversed(data))
    weights = list(reversed(weights))
    return data, weights


def sandpile(p, bins=np.geomspace(2e0, 3e3, 400),
             var="distance", comparison_pop=None, top_ax_dict={},
             inset_ax_dict={}, inset_n_bins=25,
             fig=None, axes=None, show=True):
    """Create a sandpile plot related to a population's supernovae"""
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 14))

    # set up bins and widths
    bin_centres, widths = get_centres_widths(bins)

    # setup data and weights based on variable choice
    data, weights = get_data_and_weights(p, var, widths)

    # set x-axis scale and labels, grid also
    for ax in axes:
        ax.set(xscale="log" if var != "time" else "linear",
               xlabel=xlabel[var])
        if var == "time":
            ax.set_xlim([bins[0], bins[-1]])
        else:
            ax.set_xlim([bins[0], bins[-1]])
        ax.grid(linewidth=0.5, color="lightgrey")

    # start with the top panel
    ax = axes[0]

    # plot sandpile and set ylabel
    patches = ax.hist(data, bins=bins, label=labels, stacked=True, color=colours, weights=weights)[-1];
    remove_white_stripes(patches)
    ax.set_ylabel(ylabel=top_ylabel[var])
    
    # give some extra room at the top
    ax.set_ylim(top=ax.get_ylim()[-1] * 1.1)
    
    ax.set(**top_ax_dict)
    
    if var in ["distance", "ejecta_mass"]:
        # if comparing to another population then plot the entire thing on top    
        if comparison_pop is not None:
            ax.hist(np.concatenate((comparison_pop.primary_sn_distances[~comparison_pop.sn_truly_single],
                                    comparison_pop.secondary_sn_distances)).to(u.pc).value,
                    bins=bins, histtype="step", color="black", lw=2)
        
        # add markers for the median and another axis on top for ease of reading
        for d, w, c in zip(data, weights, colours):
            med = weighted_median(d, w)
            ax.axvline(med, 0.95, 1, color=c, zorder=-1, lw=3)
            # ax.annotate(f"{med:1.1f} pc", xy=(med * 1.1, ax.get_ylim()[-1] * 0.95),
            #             va="top", ha="right", rotation=60, color=c, fontsize=0.5*fs)
            
        ax_top = ax.twiny()
        ax_top.set(xlim=ax.get_xlim(), xscale=ax.get_xscale(), xticklabels=[])
        
        # inset axis
        inset_lims = (500, 2000)
        inset_loc = [0.7, 0.5, 0.28, 0.48]
        inset_ax = ax.inset_axes(inset_loc)
        
        patches = inset_ax.hist(data, weights=weights, label=labels, color=colours, 
                                bins=np.linspace(*inset_lims, inset_n_bins), stacked=True)[-1]
        remove_white_stripes(patches)

        # for d, w, l, c in zip(data, weights, labels, colours):
        #     inset_ax.hist(d, bins=np.linspace(*inset_lims, 25), weights=w, label=l, color=c, alpha=0.1);
        #     inset_ax.hist(d, bins=np.linspace(*inset_lims, 25), weights=w, label=l, color=c, histtype="step", lw=2);

        inset_ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(250))
        inset_ax.tick_params(axis='both', labelsize=0.5 * fs)
        inset_ax.set_xlim(inset_lims)
        
        inset_ax.set(**inset_ax_dict)
    
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()

        t = ax.transLimits.inverted()
        ax.plot([inset_lims[0], 10**t.transform((inset_loc[0], 0.0))[0]],
                [0, t.transform((0, inset_loc[1]))[1]], color='darkgrey', linestyle='dotted')
        ax.plot([inset_lims[1], 10**t.transform((inset_loc[0] + inset_loc[2], 0.0))[0]],
                [0, t.transform((0, inset_loc[1]))[1]], color='darkgrey', linestyle='dotted')

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
    

    # now for the bottom panel
    ax = axes[1]
    
    # use way more bins
    bins = np.linspace(bins[0], bins[-1], 1000) if var == "time" else np.geomspace(bins[0], bins[-1], 1000)
    bin_centres, widths = get_centres_widths(bins)
    
    # we only want weights if the variable is ejecta mass
    if var != "ejecta_mass":
        weights = [np.ones_like(d) / (widths[0] * 1e6) for d in data]

    # create a bunch of histograms
    hists = np.array([np.histogram(d, weights=w, bins=bins)[0].astype(float) for d, w in zip(data, weights)])
    bottom = np.zeros_like(hists[0])
    norm = np.sum(hists)

    # normalise them
    hists /= norm

    # plot each one on top of the next
    for hist, c, l in zip(hists, colours, labels):
        ax.fill_between(bin_centres, bottom, bottom + np.sum(hist) - np.cumsum(hist), color=c, label=l, lw=0)
        bottom += np.sum(hist) - np.cumsum(hist)
    
    if comparison_pop is not None:
        d = np.concatenate((comparison_pop.primary_sn_distances[~comparison_pop.sn_truly_single],
                            comparison_pop.secondary_sn_distances)).to(u.pc).value
        hist = np.histogram(d, bins=bins)[0].astype(float)
        hist /= np.sum(hist)
        ax.plot(bin_centres, np.sum(hist) - np.cumsum(hist), color="black", zorder=5)
    
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper right", fontsize=0.8*fs)
    ax.set_ylabel(bottom_ylabel[var])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    
    # inset pie chart
    inset_loc = [0.8, 0.02, 0.2, 0.5] if var == "timee" else [0.72, 0.2, 0.2, 0.5]
    pie_ax = axes[0].inset_axes(inset_loc) if var == "timee" else axes[1].inset_axes(inset_loc)
    pie_ax.pie([np.sum(hist) for hist in hists], colors=colours)
    
    # add marker for overall median of cumulative
    if var != "time":
        overall_median = weighted_median(np.concatenate(data), np.concatenate(weights))
        ax.scatter(overall_median, 0.5, s=50, color='black')
        ax.plot([bins[0], overall_median], [0.5, 0.5], color='black')
        ax.annotate(f"{overall_median:1.1f} pc", xy=(overall_median * 1.1, 0.5), fontsize=0.7 * fs, va="center",
                    bbox=dict(boxstyle="round", color="white", pad=0.0), zorder=6)
    
    # add another non-normalised axis on the right for eject mass
    if var == "ejecta_mass":
        ax_right = ax.twinx()
        ax_right.set_ylim(top=ax.get_ylim()[1] * norm)
        ax_right.set_ylabel(r"Ejecta mass > $d \,\, [\rm M_\odot$]")

    # add a vertical line for the FIRE-2 Type-II SN stopping point and shade an area for time
    if var == "time":
        for ax in axes:
            ax.axvline(37.53, linestyle="--", color="black", lw=1)
            t = ax.transLimits.inverted()
            ax.annotate("FIRE-2 Type-II stops here", xy=(37, t.transform((0.5, 0.95))[1]),
                        rotation=90, color="black", va="top", ha="right", fontsize=0.65*fs)
            
            ax.annotate(f"~{bottom[bin_centres > 37.53][0] * 100:1.0f}% occur later",
                        xy=(39, t.transform((0.5, 0.65))[1]),
                        rotation=90, color="black", va="center", ha="left", fontsize=0.65*fs)
            # ax.axvspan(37.53, bins[-1] * 1.2, color="lightgrey", zorder=-10, alpha=0.5)
            h, l = ax.get_legend_handles_labels()
            ax.legend(h[::-1], l[::-1], fontsize=0.8*fs)
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))

    if show:
        plt.show()
    return fig, axes


def find_duplicate_supernovae(p):
    uni, counts = np.unique(p.bpp[p.bpp["evol_type"] == 15]["bin_num"], return_counts=True)
    duplicate_SN1 = uni[counts > 1]
    uni, counts = np.unique(p.bpp[p.bpp["evol_type"] == 16]["bin_num"], return_counts=True)
    duplicate_SN2 = uni[counts > 1]
    p.duplicate_sn = np.concatenate((duplicate_SN1, duplicate_SN2))
    return p.duplicate_sn


def get_relative_mass_gain(p, which_star="mass_1"):
    mass_change = p.bpp.groupby("bin_num")[which_star].diff().fillna(0.0)
    mass_change[mass_change < 0] = 0
    relative_mass_gain = mass_change.groupby(mass_change.index).sum() / p.initC[which_star]
    return relative_mass_gain


def set_sn_subpop_masks(p):
    if not hasattr(p, "duplicate_sn"):
        find_duplicate_supernovae(p)

    sn_rows = p.bpp[((p.bpp["evol_type"] == 15) | (p.bpp["evol_type"] == 16)) & ~p.bpp["bin_num"].isin(p.duplicate_sn)]
    
    primary_sn_rows = sn_rows[sn_rows["evol_type"] == 15]
    secondary_sn_rows = sn_rows[sn_rows["evol_type"] == 16]

    sn_initC = p.initC.loc[sn_rows["bin_num"]]
    truly_single_bin_nums = sn_initC[sn_initC["kstar_2"] == 15].index.values
    
    rlof_nums = p.bpp[(p.bpp["evol_type"] >= 3) & (p.bpp["evol_type"] <= 8)]["bin_num"].unique()
    rmg_1 = get_relative_mass_gain(p, 'mass_1')
    rmg_2 = get_relative_mass_gain(p, 'mass_2')
    interaction_nums = np.unique(np.concatenate((rlof_nums, rmg_1[rmg_1 > -0.05].index.values, rmg_2[rmg_2 > -0.05].index.values)))
    interaction_nums = np.concatenate((rlof_nums, p.bpp["bin_num"][(p.bpp.groupby("bin_num")["mass_1"].diff().fillna(0.0) > 0.0)
                                                  | (p.bpp.groupby("bin_num")["mass_2"].diff().fillna(0.0) > 0.0)].unique()))
    
    sn_1_sep_zero = primary_sn_rows["bin_num"].isin(primary_sn_rows["bin_num"][primary_sn_rows["sep"] == 0.0])
    sn_2_sep_zero = secondary_sn_rows["bin_num"].isin(secondary_sn_rows["bin_num"][secondary_sn_rows["sep"] == 0.0])
    
    p.sn_truly_single = primary_sn_rows["bin_num"].isin(truly_single_bin_nums)
    p.sn_1_singles = ~primary_sn_rows["bin_num"].isin(interaction_nums) & ~p.sn_truly_single & ~sn_1_sep_zero
    p.sn_2_singles = ~secondary_sn_rows["bin_num"].isin(interaction_nums) & ~sn_2_sep_zero

    p.sn_1_merger = ~(p.sn_1_singles | p.sn_truly_single) & sn_1_sep_zero
    p.sn_1 = ~(p.sn_1_singles | p.sn_truly_single) & ~sn_1_sep_zero
    
    p.sn_2_merger = ~p.sn_2_singles & sn_2_sep_zero
    p.sn_2 = ~p.sn_2_singles & ~sn_2_sep_zero
    
    print(p.sn_1_singles.sum() + p.sn_2_singles.sum(), p.sn_1.sum(), p.sn_2.sum(), p.sn_1_merger.sum() + p.sn_2_merger.sum())
    
    return p.sn_1_singles, p.sn_2_singles, p.sn_1, p.sn_1_merger, p.sn_2, p.sn_2_merger
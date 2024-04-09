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


def sn_distance_histograms(p, bins=np.geomspace(2e0, 3e3, 400), fig=None, axes=None, show=True):
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 14))

    bin_centres = np.array([(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)])
    widths = np.insert(bin_centres[1:] - bin_centres[:-1], -1, bin_centres[1] - bin_centres[0])
    
    ejecta_mass_1 = p.bpp["mass_1"].diff(-1).fillna(0.0)[(p.bpp["evol_type"] == 15) & (~p.bpp["bin_num"].isin(p.duplicate_sn))]
    ejecta_mass_2 = p.bpp["mass_2"].diff(-1).fillna(0.0)[(p.bpp["evol_type"] == 16) & (~p.bpp["bin_num"].isin(p.duplicate_sn))]

    data = [np.concatenate((p.primary_sn_distances.to(u.pc).value[p.sn_1_singles], p.secondary_sn_distances.to(u.pc).value[p.sn_2_singles])),
             p.primary_sn_distances.to(u.pc).value[p.sn_1],
             p.secondary_sn_distances.to(u.pc).value[p.sn_2],
             np.concatenate((p.primary_sn_distances.to(u.pc).value[p.sn_1_merger], p.secondary_sn_distances.to(u.pc).value[p.sn_2_merger]))]
    labels = ["Effectively Single", "Primary", "Secondary", "Merger Product"]
    colours = ["grey", plt.cm.viridis(0.4), plt.cm.viridis(0.7), "gold"]

    for ax in axes:
        ax.set(xscale="log", xlabel="SN distance from parent cluster [pc]")
        ax.grid(linewidth=0.5, color="lightgrey")

    ax = axes[0]
    ax.hist(data, bins=bins, label=labels, stacked=True, color=colours);
    # ax.legend(loc="upper left", fontsize=0.7*fs)
    ax.set_ylabel(ylabel="Number of SN")
    
    
    # inset axis
    inset_lims = (500, 2000)
    inset_loc = [0.7, 0.5, 0.28, 0.48]
    inset_ax = ax.inset_axes(inset_loc)
    
    for d, l, c in zip(data, labels, colours):
        inset_ax.hist(d, bins=np.linspace(*inset_lims, 50), label=l, color=c, alpha=0.1);
        inset_ax.hist(d, bins=np.linspace(*inset_lims, 50), label=l, color=c, histtype="step", lw=2);
    
#     inset_ax.hist(data, bins=np.linspace(100, 500, 50), label=labels, stacked=True,
#             color=["grey", plt.cm.viridis(0.4), plt.cm.viridis(0.7), "gold"]);
    # inset_ax.set_xticks([100, 500])
    inset_ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(250))
    inset_ax.set_yticks([])
    inset_ax.tick_params(axis='x', labelsize=0.5 * fs)
    inset_ax.set_xlim(inset_lims)
    
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    t = ax.transLimits.inverted()
    ax.plot([inset_lims[0], 10**t.transform((inset_loc[0], 0.0))[0]],
            [0, t.transform((0, inset_loc[1]))[1]], color='darkgrey', linestyle='dotted')
    ax.plot([inset_lims[1], 10**t.transform((inset_loc[0] + inset_loc[2], 0.0))[0]],
            [0, t.transform((0, inset_loc[1]))[1]], color='darkgrey', linestyle='dotted')
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax = axes[1]
    # ax.hist(data, bins=bins, cumulative=True, stacked=True, density=True, color=colours)

    # bottom = None
    hists = np.array([np.histogram(d, bins=bins)[0].astype(float) for d in data])
    bottom = np.zeros_like(hists[0])
    hists /= np.sum(hists)
    for hist, c, l in zip(hists, colours, labels):
        print(np.sum(hist))
        ax.bar(bin_centres, np.sum(hist) - np.cumsum(hist), color=c, bottom=bottom, width=widths, label=l)
        bottom += np.sum(hist) - np.cumsum(hist)
    # phist, bins = np.histogram(p.primary_sn_distances.to(u.pc).value, bins=bins)
    # shist, bins = np.histogram(p.secondary_sn_distances.to(u.pc).value, bins=bins)
    # hist = phist + shist
    # ax.bar(bin_centres, 1 - np.cumsum(hist) / np.sum(hist), bottom=0.5, width=widths, color="#c78ee6")
    # ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=0.7*fs)
    ax.set_ylabel(r"Fraction of SNe > $d$")

    if show:
        plt.show()
    return fig, axes


def ejecta_mass_distance_histograms(p, bins=np.geomspace(2e0, 3e3, 400), title="Fiducial", fig=None, axes=None, show=True):
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 14))

    bin_centres = np.array([(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)])
    widths = np.insert(bin_centres[1:] - bin_centres[:-1], -1, bin_centres[1] - bin_centres[0])
    
    ejecta_mass_1 = p.bpp["mass_1"].diff(-1).fillna(0.0)[(p.bpp["evol_type"] == 15) & (~p.bpp["bin_num"].isin(p.duplicate_sn))]
    ejecta_mass_2 = p.bpp["mass_2"].diff(-1).fillna(0.0)[(p.bpp["evol_type"] == 16) & (~p.bpp["bin_num"].isin(p.duplicate_sn))]

    data = [np.concatenate((p.primary_sn_distances.to(u.pc).value[p.sn_1_singles], p.secondary_sn_distances.to(u.pc).value[p.sn_2_singles])),
             p.primary_sn_distances.to(u.pc).value[p.sn_1],
             p.secondary_sn_distances.to(u.pc).value[p.sn_2],
             np.concatenate((p.primary_sn_distances.to(u.pc).value[p.sn_1_merger], p.secondary_sn_distances.to(u.pc).value[p.sn_2_merger]))]
    ejecta_mass = [np.concatenate((ejecta_mass_1[p.sn_1_singles], ejecta_mass_2[p.sn_2_singles])),
                   ejecta_mass_1[p.sn_1], ejecta_mass_2[p.sn_2], np.concatenate((ejecta_mass_1[p.sn_1_merger], ejecta_mass_2[p.sn_2_merger]))]
    labels = ["Effectively Single", "Primary", "Secondary", "Merger Product"]
    colours = ["grey", plt.cm.viridis(0.4), plt.cm.viridis(0.7), "gold"]

    for ax in axes:
        ax.set(xscale="log", xlabel=r"SN distance from parent cluster, $d \, [\rm pc]$")
        ax.grid(linewidth=0.5, color="lightgrey")

    ax = axes[0]
    ax.hist(data, weights=ejecta_mass, bins=bins, label=labels, stacked=True, color=colours);
    # ax.legend(loc="upper left", fontsize=0.7*fs)
    ax.set_ylabel(ylabel=r"Ejecta mass $[\rm M_\odot]$")
    
    
    # inset axis
    inset_lims = (500, 2000)
    inset_loc = [0.7, 0.5, 0.28, 0.48]
    inset_ax = ax.inset_axes(inset_loc)
    
    for d, w, l, c in zip(data, ejecta_mass, labels, colours):
        inset_ax.hist(d, weights=w, bins=np.linspace(*inset_lims, 50), label=l, color=c, alpha=0.1);
        inset_ax.hist(d, weights=w, bins=np.linspace(*inset_lims, 50), label=l, color=c, histtype="step", lw=2);
    
#     inset_ax.hist(data, bins=np.linspace(100, 500, 50), label=labels, stacked=True,
#             color=["grey", plt.cm.viridis(0.4), plt.cm.viridis(0.7), "gold"]);
    # inset_ax.set_xticks([100, 500])
    inset_ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(250))
    inset_ax.set_yticks([])
    inset_ax.tick_params(axis='x', labelsize=0.5 * fs)
    inset_ax.set_xlim(inset_lims)
    
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    t = ax.transLimits.inverted()
    ax.plot([inset_lims[0], 10**t.transform((inset_loc[0], 0.0))[0]],
            [0, t.transform((0, inset_loc[1]))[1]], color='darkgrey', linestyle='dotted')
    ax.plot([inset_lims[1], 10**t.transform((inset_loc[0] + inset_loc[2], 0.0))[0]],
            [0, t.transform((0, inset_loc[1]))[1]], color='darkgrey', linestyle='dotted')
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax = axes[1]
    # ax.hist(data, bins=bins, cumulative=True, stacked=True, density=True, color=colours)

    # bottom = None
    hists = np.array([np.histogram(d, weights=w, bins=bins)[0].astype(float) for d, w in zip(data, ejecta_mass)])
    bottom = np.zeros_like(hists[0])
    
    norm = np.sum(hists)
    hists /= norm
    for hist, c, l in zip(hists, colours, labels):
        print(np.sum(hist))
        ax.bar(bin_centres, np.sum(hist) - np.cumsum(hist), color=c, bottom=bottom, width=widths, label=l)
        bottom += np.sum(hist) - np.cumsum(hist)
    # phist, bins = np.histogram(p.primary_sn_distances.to(u.pc).value, bins=bins)
    # shist, bins = np.histogram(p.secondary_sn_distances.to(u.pc).value, bins=bins)
    # hist = phist + shist
    # ax.bar(bin_centres, 1 - np.cumsum(hist) / np.sum(hist), bottom=0.5, width=widths, color="#c78ee6")
    # ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=0.7*fs)
    ax.set_ylabel(r"Fraction of ejecta mass > $d$")
    
    ax_right = ax.twinx()
    ax_right.set_ylim(top=ax.get_ylim()[1] * norm)
    ax_right.set_ylabel(r"Ejecta mass > $d \,\, [\rm M_\odot$]")
    
    axes[0].set_title(title, fontsize=fs)

    if show:
        plt.show()
    return fig, axes

def sn_time_histograms(p, bins=np.linspace(0, 200, 500), fig=None, axes=None, show=True, log=False):
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 14))

    bin_centres = np.array([(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)])
    widths = np.insert(bin_centres[1:] - bin_centres[:-1], -1, bin_centres[1] - bin_centres[0])
    
    sn_time_1 = p.bpp["tphys"][(p.bpp["evol_type"] == 15) & (~p.bpp["bin_num"].isin(p.duplicate_sn))]
    sn_time_2 = p.bpp["tphys"][(p.bpp["evol_type"] == 16) & (~p.bpp["bin_num"].isin(p.duplicate_sn))]

    data = [np.concatenate((sn_time_1[p.sn_1_singles], sn_time_2[p.sn_2_singles])),
            sn_time_1[p.sn_1],
            sn_time_2[p.sn_2],
            np.concatenate((sn_time_1[p.sn_1_merger], sn_time_2[p.sn_2_merger]))]
    
    labels = ["Effectively Single", "Primary", "Secondary", "Merger Product"]
    colours = ["grey", plt.cm.viridis(0.4), plt.cm.viridis(0.7), "gold"]
    
    ax = axes[0]
    ax.hist(data, bins=bins, label=labels, stacked=True, weights=[np.ones_like(d) / (widths[0] * 1e6) for d in data], color=colours);
    ax.set_ylabel(ylabel=r"SNe Rate $\rm [yr^{-1}]$")
    # ax.set_yscale("log")
    # ax.set_ylim(bottom=7e1)
    
    ax = axes[1]
    hists = np.array([np.histogram(d, bins=bins)[0].astype(float) for d in data])
    bottom = np.zeros_like(hists[0])
    
    norm = np.sum(hists)
    hists /= norm
    for hist, c, l in zip(hists, colours, labels):
        print(np.sum(hist))
        ax.bar(bin_centres, np.sum(hist) - np.cumsum(hist), color=c, bottom=bottom, width=widths, label=l)
        bottom += np.sum(hist) - np.cumsum(hist)
    ax.set_ylabel(r"Fraction of SNe after $t$")

    for ax in axes:
        ax.set(xlim=(3 if log else -3, 203), xlabel=r"Time since cluster birth, $t \, [\rm Myr]$", xscale="log" if log else "linear")
        ax.grid(linewidth=0.5, color="lightgrey")
        ax.axvline(37.53, linestyle="--", color="black", lw=1)
        t = ax.transLimits.inverted()
        ax.annotate("FIRE-2 Type-II stops here", xy=(37, t.transform((0.5, 0.95))[1]),
                    rotation=90, color="black", va="top", ha="right", fontsize=0.65*fs)
        ax.axvspan(37.53, 220, color="lightgrey", zorder=-10, alpha=0.5)
        ax.legend(fontsize=0.6 * fs)

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


def set_sn_subpop_masks(p):
    if not hasattr(p, "duplicate_sn"):
        find_duplicate_supernovae(p)

    sn_rows = p.bpp[((p.bpp["evol_type"] == 15) | (p.bpp["evol_type"] == 16)) & ~p.bpp["bin_num"].isin(p.duplicate_sn)]
    
    primary_sn_rows = sn_rows[sn_rows["evol_type"] == 15]
    secondary_sn_rows = sn_rows[sn_rows["evol_type"] == 16]

    sn_initC = p.initC.loc[sn_rows["bin_num"]]
    truly_single_bin_nums = sn_initC[sn_initC["kstar_2"] == 15].index.values
    
    rlof_nums = p.bpp[(p.bpp["evol_type"] >= 3) & (p.bpp["evol_type"] <= 8)]["bin_num"].unique()
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
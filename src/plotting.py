import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorsys
import pandas as pd
import gala.dynamics as gd
from copy import copy

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
# colours = ["skyblue", plt.cm.viridis(0.4), plt.cm.viridis(0.7), "#DDA3E0"]
# colours = ["skyblue", plt.cm.viridis(0.4), "#67C1AA", "#EC8153"]
# colours = ["skyblue","#67C1AA", plt.cm.viridis(0.4), "#EC8153"]
colours = ["skyblue", plt.cm.viridis(0.4), "#4EB89D", "#B777D5"]


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


def sandpile(p, bins=np.geomspace(2e0, 3e3, 75),
             var="distance", comparison_pop=None, top_ax_dict={},
             inset_ax_dict={}, inset_n_bins=15, FIRE_lines=[(37.53, "FIRE-2")],
             fig=None, axes=None, show=True, save_path=None):
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

    # if comparing to another population then plot the entire thing on top    
    if comparison_pop is not None:
        comp_data, comp_weights = get_data_and_weights(comparison_pop, var, widths)
        ax.hist(np.concatenate(comp_data), weights=np.concatenate(comp_weights),
                bins=bins, histtype="step", color="black", lw=2)
    
    if var in ["distance", "ejecta_mass"]:
        
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
        for line, line_label in FIRE_lines:
            for ax in axes:
                ax.axvline(line, linestyle="dotted", color="black", lw=1)
                t = ax.transLimits.inverted()

                if len(FIRE_lines) == 1:
                    ax.annotate(f"{line_label} Type-II stops here", xy=(line - 0.5, t.transform((0.5, 0.95))[1]),
                                rotation=90, color="black", va="top", ha="right", fontsize=0.65*fs)
                    ax.annotate(f"~{bottom[bin_centres > line][0] * 100:1.0f}% occur later",
                                xy=(line + 2, t.transform((0.5, 0.65))[1]),
                                rotation=90, color="black", va="center", ha="left", fontsize=0.65*fs)
                else:
                    ax.annotate(f"{line_label}", xy=(line, t.transform((0.5, 0.979))[1]),
                                rotation=90, color="black", va="top", ha="center", fontsize=0.65*fs,
                                bbox=dict(boxstyle="round", ec="white", fc="white"))
                # ax.axvspan(37.53, bins[-1] * 1.2, color="lightgrey", zorder=-10, alpha=0.5)
                h, l = ax.get_legend_handles_labels()
                ax.legend(h[::-1], l[::-1], fontsize=0.8*fs)
                ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))

    if save_path is not None:
        plt.savefig(save_path, format="pdf" if save_path.endswith("pdf") else "png", bbox_inches="tight")
    
    if show:
        plt.show()
    return fig, axes


def _lighten_colour(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    try:
        c = mpl.colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def compare_variations(pops, pop_labels,
                       group_labels=["Fiducial", "Common-Envelope", "Mass Transfer", "Supernova kicks"],
                       group_start_inds=[1, 3, 6], annotate_loc=380, quantity="distance", combo_col="#c4c4c4",
                       fiducial_settings=None, show_rel_bars=True, show_legend=True, show_labels=True, figwidth=19, figheight=10,
                       fig=None, axes=None, show=True):

    if fig is None or axes is None:
        if show_rel_bars:
            fig, axes = plt.subplots(2, 1, figsize=(figwidth, figheight * 1.2), gridspec_kw={"height_ratios": [2, 10]})
            fig.subplots_adjust(hspace=0.0)
            ax_top, ax = axes
            ax_top.axis("off")
        else:
            fig, ax = plt.subplots(figsize=(figwidth, figheight))

    intraspacing = 0.16
    
    # determine the positions based on gaps/groups
    positions = np.arange(len(pops)).astype(float)
    for g in group_start_inds:
        positions[g:] += 0.5
    
    # add separator lines
    for g in group_start_inds:
        ax.axvline(np.mean(positions[g - 1:g + 1]) + 0.16, color="black", lw=0.5)
        
    # define the groups of positions to find the centres of each group
    groups = []
    for i in range(len(group_start_inds) - 1):
        if i == 0:
            groups.append((0, group_start_inds[i]))
        groups.append((group_start_inds[i], group_start_inds[i + 1]))
        if i == len(group_start_inds) - 2:
            groups.append((group_start_inds[i + 1], None))
            
    # label the group at its centre
    for g, l in zip(groups, group_labels):
        ax.annotate(l, xy=(np.mean(positions[g[0]:g[1]] + intraspacing), annotate_loc), fontsize=0.8*fs,
                    ha="center", bbox=dict(boxstyle="round", fc="white", ec='white'), va="top")
    
    if fiducial_settings is not None:
        ax.annotate('\n'.join(fiducial_settings),
                    xy=(np.mean(positions[groups[0][0]:groups[0][1]] + intraspacing), annotate_loc * 0.95),
                    fontsize=0.6*fs, color="grey", ha="center",
                    bbox=dict(boxstyle="round", fc="white", ec='white'), va="top")
    ax.set_ylim(top=annotate_loc * 1.025)

    global colours, labels
    colours = [colours[2], colours[1], colours[0]]
    labels = [labels[2], labels[1], labels[0]]

    fid_data = get_data_and_weights(pops["fiducial"], quantity, widths=[1])[1][:-1]
    
    for j, p in enumerate(pops):
        data, _ = get_data_and_weights(pops[p], quantity, widths=[1])
        data = list(reversed(data[:-1]))
        
        combo_data = np.concatenate(data)

        lowest = np.inf
    
        for i in range(len(data)):
            low, med, high = np.percentile(data[i], [25, 50, 75])
            ax.fill_between([positions[j] + i * intraspacing - intraspacing / 2, positions[j] + i * intraspacing + intraspacing / 2],
                            [low, low], [high, high],
                            color=colours[i])
            ax.plot([positions[j] + i * intraspacing - intraspacing / 2, positions[j] + i * intraspacing + intraspacing / 2],
                    [med, med], lw=2, color=_lighten_colour(colours[i], 1.15), zorder=10)

            lowest = min(low, lowest)

            # print(f"{len(data[i]) / len(combo_data):1.2f}")
            
            if p == 'fiducial':
                ax.axhline(np.median(data[i]), color=_lighten_colour(colours[i], 1.15),
                           zorder=-1, linestyle="-", alpha=0.5)

        if show_rel_bars:
            for i in range(len(data)):
                # ax.annotate(f"{len(data[i]) / len(combo_data):1.2f}", xy=(positions[j] + i * intraspacing, lowest),
                #             ha="center", va="top", fontsize=0.6*fs, color=_lighten_colour(colours[i], 1.15), rotation=60)
                ax_top.bar(x=positions[j] + i * intraspacing, height=len(data[i]),# / len(fid_data[i]),
                           edgecolor=colours[i], facecolor=colours[i], alpha=0.5, width=intraspacing, lw=0)
            # ax_top.bar(x=positions[j] + (i + 1) * intraspacing, height=len(combo_data) / len(np.concatenate(fid_data)), color="grey", width=intraspacing)
            
        
        comb_low, comb_med, comb_high = np.percentile(combo_data, [25, 50, 75])
        ax.fill_between([positions[j] - 0.05, positions[j] + intraspacing * 2 + 0.05],
                        [comb_low, comb_low], [comb_high, comb_high],
                        color=combo_col, zorder=-1)
        ax.plot([positions[j] - 0.1, positions[j] + intraspacing * 2 + 0.1], [comb_med, comb_med], lw=3, color=_lighten_colour(combo_col, 1.15), zorder=0)
        if p == "fiducial":
            ax.axhline(comb_med, color=combo_col, zorder=-1, linestyle="-", alpha=0.5)
            
                
    for l, c in zip(labels, colours):
        ax.axvspan(np.nan, np.nan, color=c, label=l)
    ax.axvspan(np.nan, np.nan, color=combo_col, label="Total population")

    if show_legend:
        ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.28 if show_rel_bars else 1.08), fontsize=0.8*fs)
                
    ax.set(yscale="linear", ylabel=xlabel[quantity],
           xticks=positions + intraspacing)
    ax.set_xticklabels(pop_labels if show_labels else ["" for _ in positions], rotation=45)
    ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5 + intraspacing * 2)

    if show_rel_bars:
        ax_top.set(xlim=ax.get_xlim())
    
    top_ax = ax.twinx()
    top_ax.set(ylim=ax.get_ylim(), yscale="linear")
    
    if show:
        plt.show()

    return fig, ax


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

    aic_nums = sn_rows[((sn_rows["kstar_1"] >= 10) & (sn_rows["kstar_1"] <= 12)) |
                       ((sn_rows["kstar_2"] >= 10) & (sn_rows["kstar_2"] <= 12))]["bin_num"].unique()
    p.aic = np.isin(sn_rows["bin_num"].unique(), aic_nums)

    sn_initC = p.initC.loc[sn_rows["bin_num"]]
    truly_single_bin_nums = sn_initC[sn_initC["kstar_2"] == 15].index.values
    
    rlof_nums = p.bpp[(p.bpp["evol_type"] >= 3) & (p.bpp["evol_type"] <= 8)]["bin_num"].unique()
    rmg_1 = get_relative_mass_gain(p, 'mass_1')
    rmg_2 = get_relative_mass_gain(p, 'mass_2')
    interaction_nums = np.unique(np.concatenate((rlof_nums, rmg_1[rmg_1 > -0.05].index.values, rmg_2[rmg_2 > -0.05].index.values)))
    interaction_nums = np.concatenate((rlof_nums, p.bpp["bin_num"][(p.bpp.groupby("bin_num")["mass_1"].diff().fillna(0.0) > 0.0)
                                                  | (p.bpp.groupby("bin_num")["mass_2"].diff().fillna(0.0) > 0.0)].unique()))

    kick_rows = p.kick_info.loc[secondary_sn_rows["bin_num"].unique()].drop_duplicates(subset="bin_num", keep="first")
    nonzero_ejections = kick_rows["vsys_2_total"] > 5 * u.km / u.s
    
    sn_1_sep_zero = primary_sn_rows["bin_num"].isin(primary_sn_rows["bin_num"][primary_sn_rows["sep"] == 0.0])
    sn_2_sep_zero = secondary_sn_rows["bin_num"].isin(secondary_sn_rows["bin_num"][secondary_sn_rows["sep"] == 0.0])
    
    p.sn_truly_single = primary_sn_rows["bin_num"].isin(truly_single_bin_nums)
    p.sn_1_singles = ~primary_sn_rows["bin_num"].isin(interaction_nums) & ~p.sn_truly_single & ~sn_1_sep_zero
    p.sn_2_singles = ~secondary_sn_rows["bin_num"].isin(interaction_nums) & ~sn_2_sep_zero & ~nonzero_ejections

    p.sn_1_merger = ~(p.sn_1_singles | p.sn_truly_single) & sn_1_sep_zero
    p.sn_1 = ~(p.sn_1_singles | p.sn_truly_single | primary_sn_rows["bin_num"].isin(aic_nums)) & ~sn_1_sep_zero
    
    p.sn_2_merger = ~p.sn_2_singles & sn_2_sep_zero
    p.sn_2 = ~(p.sn_2_singles | secondary_sn_rows["bin_num"].isin(aic_nums)) & ~sn_2_sep_zero
    
    print(p.sn_1_singles.sum() + p.sn_2_singles.sum(), p.sn_1.sum(), p.sn_2.sum(), p.sn_1_merger.sum() + p.sn_2_merger.sum())
    
    return p.sn_1_singles, p.sn_2_singles, p.sn_1, p.sn_1_merger, p.sn_2, p.sn_2_merger
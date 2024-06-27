from __future__ import annotations

import pathlib
from itertools import product as itprod

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import color_palette as snscolor

plt.rcParams.update(
    {
        "font.size": 9,
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": "cmr10",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
    }
)


AADR_VERSION = "v54.1.p1"
EXTRACTED_PREFIX = f"extracted/GB_{AADR_VERSION}"
SEED = 22
# all of them
ANNO_FILE = pathlib.Path(f"{EXTRACTED_PREFIX}_capture_SG_pre_pca_inds.table")


DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def getDefaultColor(idx):
    return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]


DEFAULT_MARKERS = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "X", "D"]


def getDefaultMarker(idx):
    return DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)]


def plotPCA(
    pcs,
    evals,
    categories,
    pca_plot_file,
    selectMask=None,
    plotCentroid=False,
    axtext="",
):
    color_palette = snscolor(cc.glasbey, n_colors=len(categories)).as_hex()
    # for now, otherwise we might get some problems
    assert selectMask is None

    # default: select all
    if selectMask is None:
        selectMask = np.ones(pcs.shape[0], dtype=bool)

    # plot PC 1 vs PC 2
    firstAxis = 0
    secondAxis = 1
    # firstAxis = 2
    # secondAxis = 3
    pcOne = pcs[:, firstAxis]
    pcTwo = pcs[:, secondAxis]
    (minOne, maxOne) = (pcOne[selectMask].min(), pcOne[selectMask].max())
    (minTwo, maxTwo) = (pcTwo[selectMask].min(), pcTwo[selectMask].max())
    varOne = evals[firstAxis] / np.sum(evals)
    varTwo = evals[secondAxis] / np.sum(evals)

    theMarker = np.zeros(pcs.shape[0], dtype=str)
    theColor = np.array(np.repeat("#000000", pcs.shape[0]))
    # print (theMarker)
    # print (theColor)
    theLegend = []
    legendColor = []
    legendMarker = []
    # categories = numpy.flip (categories)
    for c_i, (catMask, catName, catMarker, _) in enumerate(categories):
        catColor = color_palette[c_i]
        thisMask = catMask & selectMask
        # do we have anything to plot in this category?
        if np.sum(thisMask) > 0:
            theMarker[thisMask] = catMarker
            theColor[thisMask] = catColor
            # print (catName, catColor)
            theLegend.append(catName.replace("_CAP_", "."))
            legendColor.append(catColor)
            legendMarker.append(catMarker)

    # print (theMarker)
    # print (theColor)

    # plot points in random order
    theOrder = np.random.default_rng(SEED).permutation(np.arange(pcs.shape[0]))

    theAlpha = 0.25 if plotCentroid else 1
    for idx in theOrder:
        plt.plot(
            pcOne[idx],
            pcTwo[idx],
            theMarker[idx],
            color=theColor[idx],
            markersize=1,
            alpha=theAlpha,
        )

    # plot centroids after this
    for c_i, (catMask, _, catMarker, _) in enumerate(categories):
        catColor = color_palette[c_i]
        thisMask = catMask & selectMask
        # do we have anything to plot in this category?
        if np.sum(thisMask) > 0 and plotCentroid:
            centOne = np.mean(pcOne[thisMask])
            centTwo = np.mean(pcTwo[thisMask])
            plt.plot(
                centOne, centTwo, catMarker, color=catColor, markersize=3.75, alpha=1
            )

    plt.legend(
        theLegend,
        fontsize=7,
        labelspacing=0.2,
        handlelength=1.5,
        handleheight=0.5,
        handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,
    )
    ax = plt.gca()
    ax.text(-0.2, 0.97, rf"$\bf{{{axtext}}}$", fontsize=13, transform=ax.transAxes)
    leg = ax.get_legend()
    for i in np.arange(len(legendColor)):
        leg.legend_handles[i].set_color(legendColor[i])
        leg.legend_handles[i].set_marker(legendMarker[i])
        leg.legend_handles[i].set_alpha(1)
        leg.legend_handles[i].set_markersize(4)

    plt.xlabel(f"PC1 ({varOne*100:.2f}%)")
    plt.ylabel(f"PC2 ({varTwo*100:.2f}%)")

    # coord grid
    if minOne < 0 < maxOne:
        plt.vlines(0, minTwo, maxTwo, linestyles="--", colors="black", lw=0.5)
    if minTwo < 0 < maxTwo:
        plt.hlines(0, pcOne.min(), pcOne.max(), linestyles="--", colors="black", lw=0.5)

    plt.savefig(pca_plot_file, bbox_inches="tight")
    plt.clf()


def allPcaPlots():
    # make plots a bit bigger for now

    # we need the publications for refined labels
    anno = pd.read_csv(ANNO_FILE, sep="\t", low_memory=False)
    # compress names of publications
    short_pubs = np.array(
        [x.split()[0][0] + x.split()[0][-2:] for x in anno["Publication"]]
    )

    # go through all combinations
    # for reference in ['broad', 'europe']:
    for reference, to_shrink, genotyping in itprod(
        ["broad", "europe", "gbr_ceu"],
        ["shrinkage", "noshrinkage"],
        ["capture_only", "capture_SG"],
    ):
        axtext = ""
        if to_shrink == "shrinkage" and genotyping == "capture_only":
            if reference == "broad":
                axtext = "A"
            elif reference == "europe":
                axtext = "B"
        plt.figure(figsize=(3.1, 3.1), layout="constrained")
        # load specific pca files
        eval_file = pathlib.Path(
            f"{EXTRACTED_PREFIX}_{genotyping}_{reference}_pca_{to_shrink}.eval"
        )
        evec_file = pathlib.Path(
            f"{EXTRACTED_PREFIX}_{genotyping}_{reference}_pca_{to_shrink}.evec"
        )

        evals = np.array(pd.read_csv(eval_file, header=None)).flatten()
        evecFrame = pd.read_csv(evec_file, sep=r"\s+")

        # get the data in convenient format
        pcs = np.array(evecFrame.iloc[:, :-1])
        popLabels = np.array(evecFrame.iloc[:, -1])
        ids = np.array(evecFrame.index)

        # leverage pub labels from just ancients to whole pca sample
        tmp_anno_ids = list(anno["Genetic_ID"])
        leveraged_pubs = np.array(
            [
                "" if (x not in tmp_anno_ids) else (short_pubs[tmp_anno_ids.index(x)])
                for x in ids
            ]
        )

        # have some categories for plotting
        categories = []

        # publication categories
        colorIdx = 0
        for pop in sorted(set(popLabels)):
            thisPopMask = popLabels == pop
            if pop in ["FOCAL_ANCIENT_SG", "FOCAL_ANCIENT_CAPTURE"]:
                short_pop = "ANC_CAP" if "CAPTURE" in pop else "ANC_SG"
                # special
                for pub in set(short_pubs):
                    thisPubMask = leveraged_pubs == pub
                    if (thisPopMask & thisPubMask).sum() > 0:
                        categories.append(
                            (
                                thisPopMask & thisPubMask,
                                short_pop + "_" + pub,
                                getDefaultMarker(colorIdx),
                                getDefaultColor(colorIdx),
                            )
                        )
                        colorIdx += 1
            else:
                # regular
                categories.append(
                    (
                        thisPopMask,
                        pop,
                        getDefaultMarker(colorIdx),
                        getDefaultColor(colorIdx),
                    )
                )
                colorIdx += 1

        # plot a pca with all samples
        pca_plot_file = pathlib.Path(
            f"{EXTRACTED_PREFIX}_{genotyping}_{reference}_pca_{to_shrink}.pdf"
        )
        plotCentroid = reference != "broad"
        plotPCA(
            pcs,
            evals,
            categories,
            pca_plot_file,
            plotCentroid=plotCentroid,
            axtext=axtext,
        )

        # # and for the broad analysis, we also plot a zoom on ancient + europe
        # if (reference == 'broad'):
        #     zoomed_pops = ['FOCAL_ANCIENT', 'CEU.SG', 'GBR.SG', 'FIN.SG', 'TSI.SG', 'IBS.SG']
        #     selectMask = numpy.array ([x in zoomed_pops for x in popLabels])
        #     # try to omit outliers (certainly for the zoom)
        #     selectMask[(pcs[:,0] > 0) & (popLabels == 'FOCAL_ANCIENT')] = False
        #     print (ids[(pcs[:,0] > 0) & (popLabels == 'FOCAL_ANCIENT')])
        #     pca_plot_file = pathlib.Path (f"{EXTRACTED_PREFIX}_{reference}_pca_{to_shrink}_zoom.pdf")
        #     plotPCA (f"PCA: {reference}, {to_shrink}, zoom", pcs, evals, categories, pca_plot_file, selectMask=selectMask)


def main():
    # make some plots
    allPcaPlots()


if __name__ == "__main__":
    main()

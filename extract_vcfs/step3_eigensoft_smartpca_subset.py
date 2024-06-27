from __future__ import annotations

import pathlib
import subprocess

AADR_VERSION = "v54.1.p1"
EXTRACTED_PREFIX = f"extracted/GB_{AADR_VERSION}"
REFERENCE_POPS = {
    "gbr_ceu": ["GBR.SG", "CEU.SG"],
    "europe": ["GBR.SG", "FIN.SG", "IBS.SG", "TSI.SG", "CEU.SG"],
    "broad": [
        "GBR.SG",
        "FIN.SG",
        "IBS.SG",
        "TSI.SG",
        "CEU.SG",
        "YRI.SG",
        "LWK.SG",
        "CHB.SG",
        "JPT.SG",
    ],
}
ANCIENT_VERSIONS = {
    "capture_only": ["FOCAL_ANCIENT_CAPTURE"],
    "capture_SG": ["FOCAL_ANCIENT_CAPTURE", "FOCAL_ANCIENT_SG"],
}
# EITHER RUN DIRECTLY IN SHELL OR ONLY PRINT TO RUN LATER
RUN_COMMANDS = True


def performPca():
    # iterate through references
    for thisLabel, thisReference in REFERENCE_POPS.items():
        # iterate through ancient versions
        for thisAncientVersion in ANCIENT_VERSIONS:
            # specific files
            mockPrefix = f"{EXTRACTED_PREFIX}_{thisAncientVersion}_{thisLabel}"
            pca_geno_file = pathlib.Path(f"{mockPrefix}_pca.geno")
            pca_snp_file = pathlib.Path(f"{mockPrefix}_pca.snp")
            pca_ind_file = pathlib.Path(f"{mockPrefix}_pca.ind")

            # prepare the reference population file
            reference_file = pathlib.Path(f"{mockPrefix}_reference.pops")
            with pathlib.Path(reference_file).open("w", encoding="locale") as ofs:
                for pop in thisReference:
                    ofs.write(pop + "\n")

            # to shrink or not to shrink
            for to_shrink in ["shrinkage", "noshrinkage"]:
                # even more specific files
                evec_out_file = pathlib.Path(f"{mockPrefix}_pca_{to_shrink}.evec")
                eval_out_file = pathlib.Path(f"{mockPrefix}_pca_{to_shrink}.eval")
                smartpca_param_file = pathlib.Path(
                    f"{mockPrefix}_smartpca_{to_shrink}.param"
                )

                # prepare the parameter string for smartpca
                paramString = (
                    f"indivname: {pca_ind_file}\n"
                    + f"snpname: {pca_snp_file}\n"
                    + f"genotypename: {pca_geno_file}\n"
                    + f"evecoutname: {evec_out_file}\n"
                    + f"evaloutname: {eval_out_file}\n"
                    +
                    ## this does some ld pruning
                    "killr2: YES\n"
                    + "r2thresh: 0.1\n"
                    + "lsqproject: YES\n"
                    +
                    ## TWO more decimal places!!!!!
                    "hiprecision: YES\n"
                    +
                    ## remove outliers
                    "numoutlieriter: 5\n"
                    +
                    ## default is 10
                    # "numoutlierevec: 10\n" + \
                    ## default YES, so leave it at that
                    # "altnormstyle: NO\n" + \
                    ## this is the default
                    # "numoutevec: 10\n" + \
                    ## YES here would disable all the things that make PCA work
                    "fastmode: NO\n"
                    +
                    ## we have 1000 genomes populations of size each roughly 100, should be weighted equally
                    # "popsizelimit: 25\n"
                    f"poplistname: {reference_file}\n"
                )

                # shrinkage options or not
                if to_shrink == "shrinkage":
                    paramString += "shrinkmode: YES\n" + "newshrink: YES\n"
                elif to_shrink == "noshrinkage":
                    paramString += "shrinkmode: NO\n" + "newshrink: NO\n"
                else:
                    # tertium non datur
                    raise AssertionError()

                # write the param file
                with pathlib.Path(smartpca_param_file).open(
                    "w", encoding="locale"
                ) as ofs:
                    ofs.write(paramString)

                # and run smartpca
                smartpcaCmd = f"smartpca -p {smartpca_param_file}"
                if RUN_COMMANDS:
                    subprocess.run([smartpcaCmd], shell=True, text=True, check=False)


def main():
    performPca()


if __name__ == "__main__":
    main()

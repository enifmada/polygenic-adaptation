from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from subprocess import run as sprun


def wget_command(pheno_str, output_dir, output_fname):
    pheno_file = Path(output_dir) / output_fname
    if not pheno_file.is_file():
        sprun(["wget", pheno_str, "-O", pheno_file], check=False)
    return 1


def main():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", help="output BASE directory")
    parser.add_argument("--pheno_loc_file", help="txt of files to wget")
    parser.add_argument("--pheno_list_file", help="txt of phenotype file names")

    smk = vars(parser.parse_args())
    with Path(smk["pheno_list_file"]).open("r") as f:
        phenos = f.readlines()

    phenos = [p.strip() for p in phenos]
    pheno_codes = [p.partition(".")[0] for p in phenos]
    phenos_simple = [p + ".tsv.bgz" for p in pheno_codes]

    with Path(smk["pheno_loc_file"]).open("r") as f:
        pheno_locs = f.readlines()

    pheno_locs = [p.strip() for p in pheno_locs]

    for f_i, pheno_str in enumerate(pheno_locs):
        wget_command(pheno_str, smk["output_dir"], phenos_simple[f_i])


if __name__ == "__main__":
    main()

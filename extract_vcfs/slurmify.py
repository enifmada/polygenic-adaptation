from __future__ import annotations

import pathlib
import sys

TMP_DIR = "slurmify_tmp"
NUM_CORES = 1
NUM_GB = 16


def readCommands(commandFile):
    # read file
    with pathlib.Path(commandFile).open() as ifs:
        theLines = ifs.readlines()

    # go through and clean
    cleaned = []
    for thisLine in theLines:
        cleaned.append(thisLine.strip())

    return cleaned


def prepareSlurmScripts(commandList, metaSubmitScript, tmpDir):
    # set up the tmp dir
    pathlib.Path(tmpDir).mkdir(parents=True)

    # set up meta script
    with pathlib.Path(metaSubmitScript).open("w") as metafs:
        # go through commands
        for cmdIdx, thisCmd in enumerate(commandList):
            # generic name
            thisName = f"slurm{cmdIdx}"
            thisScriptName = pathlib.Path(f"{tmpDir}/{thisName}.sbatch")
            with pathlib.Path(thisScriptName).open("w") as ofs:
                # needs to be in script
                ofs.write("#!/bin/bash\n")
                # give it a name
                ofs.write(f"#SBATCH --job-name={thisName}\n")
                # only 1 node
                ofs.write("#SBATCH --nodes=1\n")
                # only 1 task
                ofs.write("#SBATCH --ntasks=1\n")
                # we want this many cpus
                ofs.write(f"#SBATCH --cpus-per-task={NUM_CORES}\n")
                # also a bit of time
                ofs.write("#SBATCH --time=12:00:00\n")
                # some memory
                ofs.write(f"#SBATCH --mem={NUM_GB}gb\n")
                # output
                ofs.write(f"#SBATCH --output={tmpDir}/%x.o%j\n")
                ofs.write(f"#SBATCH --error={tmpDir}/%x.e%j\n")
                # and finally the command
                ofs.write(thisCmd + "\n")

            # add something to a meta script
            metafs.write("sbatch %s\n" % thisScriptName)


def main():
    # need exactly one additional argument
    assert len(sys.argv) == 2

    # get the commands
    commandList = readCommands(sys.argv[-1])

    prepareSlurmScripts(commandList, "metasubmit.sh", TMP_DIR)


if __name__ == "__main__":
    main()

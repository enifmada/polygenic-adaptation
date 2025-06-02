from __future__ import annotations

from pathlib import Path

###### MODIFY
qsub_dir = Path('qsubs')
max_run_hours = 1
num_cores = 1
huge_mem_required = 50
large_mem_required = 20
small_mem_required = 8

###### DO NOT MODIFY

def writeQsubs():
    hmm_cmds = []
    # get a meta file going
    script_file = Path("meta_test_EM.sh")
    with Path.open(script_file, "w") as file:
        hmm_cmd = "python3 ukbb_ashr_testing.py"
        sbatchfile = qsub_dir/"testing.sbatch"
        sbatchOutFile = qsub_dir/"testing.out"
        sbatchErrFile = qsub_dir/"testing.err"
        with Path.open(sbatchfile, 'w') as qsubScriptFile:
            qsubScriptFile.write("#!/bin/bash\n" + \
                                 "#SBATCH -J ashtest\n" + \
                                 f"#SBATCH --time={max_run_hours}:00:00\n" + \
                                 f"#SBATCH --cpus-per-task={num_cores}\n" + \
                                 f"#SBATCH --mem={large_mem_required}gb\n" + \
                                 f"#SBATCH -o {sbatchOutFile}\n" + \
                                 f"#SBATCH -e {sbatchErrFile}\n" + \
                                 f"{hmm_cmd}\n")
                    # and add to meta file
        file.write(f"sbatch {sbatchfile}\n")
        hmm_cmds.append(hmm_cmd)


def main():
    writeQsubs()

if __name__ == "__main__":
    main()
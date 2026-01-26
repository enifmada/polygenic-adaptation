import json
from pathlib import Path
from numpy import loadtxt as nploadtxt

base_dir = "../../../polyoutput/slim_testing/"+config["file_prefix"][:-1]
Path(base_dir).mkdir(parents=False, exist_ok=True)

true_base_dir = base_dir + "/"
subdir_strs = ["betas", "slims", "data", "trajs"]
subdir_singles = ["betas", "slim", "data", "trajs", "grid", "surface"]
subdir_ftypes = [".txt", ".txt", ".csv", ".pdf", ".csv", ".pdf"]

if "end_gen" in config and "end_ns" in config:
    cond_str = f"--condition_on_seg --end_gen {config['end_gen']} --end_ns {config['end_ns']} "
    cond_short_str = "cond_"
    COND_FLAG = True
    assert config['end_gen'] > config['num_gens']
else:
    cond_str = ""
    cond_short_str = ""
    COND_FLAG = False

if COND_FLAG:
    subdir_strs.extend(["grids_cond", "surfaces_cond"])
    surface_str = "surfaces_cond"
    grids_str = "grids_cond"
else:
    subdir_strs.extend(["grids", "surfaces"])
    surfaces_str = "surfaces"
    grids_str = "grids"

subdirs = [true_base_dir+subdir+"/" for subdir in subdir_strs]
subdir_str_dict = {}
final_output = []
add_w_lineplot = False
for s_i, subdir in enumerate(subdirs):
    subdir_str_dict[subdir_strs[s_i]] = subdir
    Path(subdir[:-1]).mkdir(parents=True, exist_ok=True)
    temp_w_strs = expand(subdir+config['file_prefix']+"w{w}_s{s}_"+subdir_singles[s_i]+subdir_ftypes[s_i], w=config["omega_array"], s=range(config["num_replicates"]))
    if "surface" not in subdir and "betas" not in subdir:
        pass
        #final_output.extend(temp_w_strs)
    if s_i == 0 and len(temp_w_strs)>0:
        add_w_lineplot = True

if add_w_lineplot:
    if config["analysis_betas"] == "gwas":
        #final_output.append(true_base_dir+config['file_prefix']+"w_lineplot_truebetas.pdf")
        #final_output.append(true_base_dir+config['file_prefix']+"w_lineplot_gwasbetas.pdf")
        # final_output.append(true_base_dir+config['file_prefix']+f"w_{cond_short_str}truebetas_boxplot.pdf")
        # final_output.append(true_base_dir+config['file_prefix']+f"w_{cond_short_str}gwasbetas_boxplot.pdf")
        # final_output.append(true_base_dir+config['file_prefix']+f"w_{cond_short_str}Serrplot.pdf")
        # final_output.append(true_base_dir+config['file_prefix']+f"w_{cond_short_str}werrplot.pdf")
        # final_output.append(true_base_dir+config['file_prefix']+f"w_{cond_short_str}zscplot.pdf")
        # final_output.append(true_base_dir+config['file_prefix']+f"w_{cond_short_str}zerrplot.pdf")
        final_output.append(true_base_dir+config['file_prefix']+f"w_un{cond_short_str}truebetas_presboxplot.pdf")
        final_output.append(true_base_dir+config['file_prefix']+f"w_un{cond_short_str}gwasbetas_presboxplot.pdf")
        final_output.append(true_base_dir+config['file_prefix']+f"w_un{cond_short_str}Serrplot.pdf")
        final_output.append(true_base_dir+config['file_prefix']+f"w_un{cond_short_str}werrplot.pdf")
        final_output.append(true_base_dir+config['file_prefix']+f"w_un{cond_short_str}zscplot.pdf")
        final_output.append(true_base_dir+config['file_prefix']+f"w_un{cond_short_str}zerrplot.pdf")
    else:
        final_output.append(true_base_dir+config['file_prefix']+f"w_{cond_short_str}lineplot.pdf")

print(final_output)

if "hmm_Ne" not in config:
    config["hmm_Ne"] = config["Ne"]

if "ld_output" not in config:
    config["ld_output"] = -1

if "h2" not in config:
    config["h2"] = 1

if "init_freq" in config:
    config["freq_init"] = config["init_freq"]

if "analysis_betas" not in config:
    config["analysis_betas"] = "ground_truth"

if config["num_loci"] == "all":
    betasfile = nploadtxt(true_base_dir+config['beta_file'])
    config["num_loci"] = betasfile.shape[0]

if "boxplot_letters" not in config:
    config["boxplot_letters"] = ["", ""]
    bpl_string = ""
else:
    bpl_string = f"--boxplot_letters {config['boxplot_letters'][0]} {config['boxplot_letters'][1]} "

if config["analysis_betas"] == "gwas":
    #need the ld output to do a gwas
    if config["ld_output"] < 0:
        config["ld_output"] = 5
    temp_betas_w_strs = expand(subdir_str_dict["betas"]+config['file_prefix']+"w{w}_s{s}_betas_gwas.txt", w=config["omega_array"], s=range(config["num_replicates"]))
    #final_output.extend(temp_betas_w_strs)

if config["ld_output"] > 0:
    temp_vcf_w_strs = expand(subdir_str_dict["slims"]+config['file_prefix']+"w{w}_s{s}_allgenos.vcf", w=config["omega_array"], s=range(config["num_replicates"]))
    #final_output.extend(temp_vcf_w_strs)
    temp_pheno_w_strs = expand(subdir_str_dict["slims"]+config['file_prefix']+"w{w}_s{s}_phenotypes.txt", w=config["omega_array"], s=range(config["num_replicates"]))
    #final_output.extend(temp_pheno_w_strs)

gen_freq_str = ""
gen_beta_str = ""
true_freq_path = ""
true_beta_path = ""
for k in config:
    if k.startswith("freq"):
        if k != "freq_mode":
            if k.endswith("file"):
                gen_freq_str += f"--{k} {true_base_dir}{config[k]} "
                true_freq_path += f"{true_base_dir}{config[k]}"
            else:
                gen_freq_str += f"--{k} {config[k]} "
    elif k.startswith("beta_"):
        if k != "beta_mode":
            if k.endswith("file"):
                gen_beta_str += f"--{k} {true_base_dir}{config[k]} "
                true_beta_path += f"{true_base_dir}{config[k]}"
            else:
                gen_beta_str += f"--{k} {config[k]} "

print(f"gbs: {gen_beta_str}")
touch_str = "touch.file" if config["analysis_betas"] == "gwas" else "touch_grass.file"

with open(true_base_dir+config['beta_file'], "r") as f:
    betas = f.readlines()

num_loci = len(betas)
mem_req = int(400000 * (num_loci/2500)**2)
if mem_req > 32000:
    tier_str = "tier3q"
elif mem_req > 8000:
    tier_str = "tier2q"
else:
    tier_str = "tier1q"

def run_slims_function(wildcards):
    return f'slim -s {int(int(wildcards.s)+1e6*float(wildcards.w))} -d beta_file="\'{true_base_dir+config['beta_file']}\'" -d freq_file="\'{true_base_dir+config['freq_file']}\'" -d omega={wildcards.w} -d dz={config["dz"]} -d h2={config["h2"]} -d num_gens={config["num_gens"]} -d num_loci={config["num_loci"]} -d Ne={config["Ne"]} -d ld_output={config["ld_output"]} -d mode="\'{config["mode"]}\'" -d output_path="\'{subdir_str_dict["slims"]}{config["file_prefix"]}w{wildcards.w}_s{wildcards.s}\'" first_slim_script.slim'

def run_sample_slims_function(wildcards, input, output):
    return f"python sample_slim.py -i {input} -o {output} --sampling_scheme {config["sampling_scheme"]} --sampling_table_file {true_base_dir+config["sampling_table_file"]} --betas_file {true_base_dir+config['beta_file']} --seed {int(int(wildcards.s)+1e6*float(wildcards.w))}"

rule all:
    input:
        final_output

if config["analysis_betas"] == "gwas":
    rule perform_gwaslim:
        input:
            subdir_str_dict["slims"]+config["file_prefix"]+"w{w}_s{s}_allgenos.vcf",
            subdir_str_dict["slims"]+config["file_prefix"]+"w{w}_s{s}_phenotypes.txt"
        output:
            subdir_str_dict["betas"]+config["file_prefix"]+"w{w}_s{s}_betas_gwas.txt"
        resources:
            slurm_partition="tier2q",
            mem_mb=10000
        shell:
            f"python perform_gwaslim.py -p {{input[1]}} -g {{input[0]}} -o {{output}}"

    rule touch_gwas:
        input:
            expand(subdir_str_dict["betas"]+config["file_prefix"]+"w{w}_s{s}_betas_gwas.txt", w=config["omega_array"], s=range(config["num_replicates"]))
        output:
            touch(true_base_dir+"touch.file")
else:
    rule dont_perform_gwaslim:
        output:
            touch(true_base_dir+"touch_grass.file")

rule sample_data:
    input:
        subdir_str_dict["slims"]+config['file_prefix']+"w{w}_s{s}_slim.txt",
    output:
        subdir_str_dict["data"]+config["file_prefix"]+"w{w}_s{s}_data.csv",
        subdir_str_dict["trajs"]+config['file_prefix']+"w{w}_s{s}_trajs.pdf"
    run:
        a = run_sample_slims_function(wildcards, input, output)
        print(a)
        shell(a)

rule run_slims:
    output:
        subdir_str_dict["slims"]+config["file_prefix"]+"w{w}_s{s}_slim.txt",
        subdir_str_dict["slims"]+config["file_prefix"]+"w{w}_s{s}_allgenos.vcf" if config["ld_output"]>0 else "",
        subdir_str_dict["slims"]+config["file_prefix"]+"w{w}_s{s}_phenotypes.txt" if config["ld_output"]>0 else ""
    #turn this into some kind of adaptive scaling thing, shouldn't be too hard. not every slim needs 100GB, some need more.
    resources:
        mem_mb=mem_req,
        slurm_partition=tier_str
    run:
        a = run_slims_function(wildcards)
        print(a)
        shell(a)


rule run_grids:
    input:
        subdir_str_dict["data"]+config['file_prefix']+"w{w}_s{s}_data.csv"
    output:
        subdir_str_dict[grids_str]+config['file_prefix']+"w{w}_s{s}_grid.csv",
        subdir_str_dict[grids_str]+config['file_prefix']+"w{w}_s{s}_grid_uncon.csv"
    #this too.
    resources:
        threads=8,
        slurm_partition="tier1q",
        mem_mb = 20000
    shell:
        f"python run_hmm.py {{input}} {{output[0]}} --time_after_zero -hs {config['hidden_states']} -sid {config['hmm_init_dist']}"
        f" {cond_str}--grid_s_max {config['grid_s_max']} -np {config['num_half_grid_points']} -Ne {config['hmm_Ne']} --progressbar --snakemake -nc {{resources.threads}}"

# rule analyze_w_grids_regression:
#     input:
#         expand(subdir_str_dict[grids_str]+config['file_prefix']+"w{w}_s{s}_grid.csv", w=config["omega_array"], s=range(config["num_replicates"])),
#         true_base_dir+touch_str
#     output:
#         true_base_dir+config['file_prefix']+f"w_{cond_short_str}truebetas_boxplot.pdf",
#         true_base_dir+config['file_prefix']+f"w_{cond_short_str}gwasbetas_boxplot.pdf",
#         true_base_dir+config['file_prefix']+f"w_{cond_short_str}Serrplot.pdf",
#         true_base_dir+config['file_prefix']+f"w_{cond_short_str}werrplot.pdf",
#         true_base_dir+config['file_prefix']+f"w_{cond_short_str}zscplot.pdf",
#         true_base_dir+config['file_prefix']+f"w_{cond_short_str}zerrplot.pdf",
#     resources:
#         mem_mb=5000
#     shell:
#         f"python analyze_fullmatched_sim.py -m {config['mode']} --vary omega -dz {config['dz']} -h2 {config['h2']} --global_vg {true_base_dir+config['beta_file']} {true_base_dir+config['freq_file']} --beta_file {true_base_dir+config['beta_file']} -i {{input}} -o {{output}}"#input

rule analyze_w_grids_uncon:
    input:
        expand(subdir_str_dict[grids_str]+config['file_prefix']+"w{w}_s{s}_grid_uncon.csv", w=config["omega_array"], s=range(config["num_replicates"])),
        true_base_dir+touch_str
    output:
        true_base_dir+config['file_prefix']+f"w_un{cond_short_str}truebetas_presboxplot.pdf",
        true_base_dir+config['file_prefix']+f"w_un{cond_short_str}gwasbetas_presboxplot.pdf",
        true_base_dir+config['file_prefix']+f"w_un{cond_short_str}Serrplot.pdf",
        true_base_dir+config['file_prefix']+f"w_un{cond_short_str}werrplot.pdf",
        true_base_dir+config['file_prefix']+f"w_un{cond_short_str}zscplot.pdf",
        true_base_dir+config['file_prefix']+f"w_un{cond_short_str}zerrplot.pdf",
    resources:
        mem_mb=5000
    shell:
        f"python analyze_fullmatched_slim.py -m {config['mode']} --vary omega -dz {config['dz']} -h2 {config['h2']} --global_vg {true_base_dir+config['beta_file']} {true_base_dir+config['freq_file']} --beta_file {true_base_dir+config['beta_file']} {bpl_string}-i {{input}} -o {{output}}"#input

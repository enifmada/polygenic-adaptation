import json
from pathlib import Path
from enum import Enum

#TODO: rewrite all omega-centric code in terms of V_S, more fundamental
# anything with omega_array -> vs_array so it's backwards compatible
# also make sure the output filenames reflect w_ vs vs_

base_dir = "../../../polyoutput/slim_testing/"+config["file_prefix"][:-1]
Path(base_dir).mkdir(parents=True, exist_ok=True)

true_base_dir = base_dir + "/"
subdir_strs = ["betas", "freqs", "sims", "data", "trajs"]
subdir_singles = ["betas", "freqs", "sim", "data", "trajs", "grid", "surface"]
subdir_ftypes = [".txt", ".txt", ".npz", ".csv", ".pdf", ".csv", ".pdf"]

class Subdir(Enum):
    BETAS = 0
    FREQS = 1
    SIMS = 2
    DATA = 3
    TRAJS = 4
    GRIDS = 5
    SURFACES = 6

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

if "sim_source" not in config:
    config["sim_source"] = "slim"

if config["sim_source"] == "slim":
    subdir_ftypes[Subdir.SIMS.value] = ".txt"
    sim_ending = phenos_ending = ".txt"
    genos_ending = ".vcf"
    SLIM_FLAG = True
else:
    sim_ending = genos_ending = phenos_ending = ".npz"
    SLIM_FLAG = False

if config["num_loci"] == "all":
    config["freq_mode"] = "matched"
    config["beta_mode"] = "matched"
    assert "beta_file" in config
    with open(true_base_dir+config['beta_file'], "r") as f:
        betas = f.readlines()

    num_loci = len(betas)
else:
    num_loci = config["num_loci"]

if "omega_array" in config:
    OMEGA_GEN = True
    config["vs_array"] = [val**2 for val in config["omega_array"]]
    wabbr = "w"
    config["omegasomething_array"] = config["omega_array"]
    analysis_ostr = "--use_omega"
    vary_str = "omega"
else:
    OMEGA_GEN = False
    wabbr = "vs"
    config["omegasomething_array"] = config["vs_array"]
    analysis_ostr = "--use_vs"
    vary_str = "vs"

subdirs = [true_base_dir+subdir+"/" for subdir in subdir_strs]
print(subdirs)
subdir_str_dict = {}
final_output = []
for s_i, subdir in enumerate(subdirs):
    subdir_str_dict[subdir_strs[s_i]] = subdir
    if config["freq_mode"] != "matched" or subdir_strs[s_i] != "freqs":
        Path(subdir[:-1]).mkdir(parents=True, exist_ok=True)
        if config["freq_mode"] != "matched" or subdir_strs[s_i] != "betas":
            temp_o_strs = expand(subdir+config['file_prefix']+f"{wabbr}{{omegasomething}}_s{{s}}_"+subdir_singles[s_i]+subdir_ftypes[s_i], omegasomething=config["omegasomething_array"], s=range(config["num_replicates"]))
            if "surface" not in subdir:
                final_output.extend(temp_o_strs)


final_output.append(true_base_dir + config['file_prefix'] + "regression.parquet")
final_output.append(true_base_dir + config['file_prefix'] + "ll.parquet")
output_plot_suffixes = []
output_plot_suffixes.append(f"{config['file_prefix']}{cond_short_str}truebetas_boxplot.pdf")
output_plot_suffixes.append(f"{config['file_prefix']}{cond_short_str}gwasbetas_boxplot.pdf")
output_plot_suffixes.append(f"{config['file_prefix']}{cond_short_str}Serrplot.pdf")
output_plot_suffixes.append(f"{config['file_prefix']}{cond_short_str}werrplot.pdf")
output_plot_suffixes.append(f"{config['file_prefix']}{cond_short_str}zscplot.pdf")
output_plot_suffixes.append(f"{config['file_prefix']}{cond_short_str}zerrplot.pdf")
output_plots = [true_base_dir + s for s in output_plot_suffixes]
final_output.extend(output_plots)

output_plot_suffixes_ll = []
output_plot_suffixes_ll.append(f"{config['file_prefix']}{cond_short_str}truebetas_boxplot_ll.pdf")
output_plot_suffixes_ll.append(f"{config['file_prefix']}{cond_short_str}gwasbetas_boxplot_ll.pdf")
output_plot_suffixes_ll.append(f"{config['file_prefix']}{cond_short_str}Serrplot_ll.pdf")
output_plot_suffixes_ll.append(f"{config['file_prefix']}{cond_short_str}werrplot_ll.pdf")
output_plot_suffixes_ll.append(f"{config['file_prefix']}{cond_short_str}zscplot_ll.pdf")
output_plot_suffixes_ll.append(f"{config['file_prefix']}{cond_short_str}zerrplot_ll.pdf")
output_plots_ll = [true_base_dir + s for s in output_plot_suffixes_ll]
final_output.extend(output_plots_ll)


if "hmm_Ne" not in config:
    config["hmm_Ne"] = config["Ne"]

if "ld_output" not in config:
    config["ld_output"] = -1

if "h2" not in config:
    config["h2"] = 1.0

if "init_freq" in config:
    config["freq_init"] = config["init_freq"]

if "analysis_betas" not in config:
    config["analysis_betas"] = "ground_truth"

if "target_vP" not in config:
    config["target_vP"] = 1.0

if "big_gwas" in config:
    big_gwas_sim_str = "--big_gwas "
    big_gwas_name_str = "big"
else:
    big_gwas_sim_str = ""
    big_gwas_name_str = ""

if "boxplot_letters" not in config:
    config["boxplot_letters"] = ["", ""]
    bpl_string = ""
else:
    bpl_string = f"--boxplot_letters {config['boxplot_letters'][0]} {config['boxplot_letters'][1]} "

if "scale_factor" not in config:
    config["scale_factor"] = 1

if config["analysis_betas"] == "gwas":
    #need the ld output to do a gwas
    if config["ld_output"] < 0:
        config["ld_output"] = 5
    temp_betas_o_strs = expand(subdir_str_dict["betas"]+config['file_prefix']+f"{wabbr}{{omegasomething}}_s{{s}}_betas_gwas.txt",
        omegasomething=config["omegasomething_array"], s=range(config["num_replicates"]))
    final_output.extend(temp_betas_o_strs)

if config["ld_output"] > 0:
    temp_vcf_o_strs = expand(subdir_str_dict["sims"]+config['file_prefix']+f"{wabbr}{{omegasomething}}_s{{s}}_{big_gwas_name_str}allgenos{genos_ending}", omegasomething=config["omegasomething_array"], s=range(config["num_replicates"]))
    temp_pheno_o_strs = expand(subdir_str_dict["sims"]+config['file_prefix']+f"{wabbr}{{omegasomething}}_s{{s}}_{big_gwas_name_str}phenotypes{phenos_ending}", omegasomething=config["omegasomething_array"], s=range(config["num_replicates"]))
    final_output.extend(temp_vcf_o_strs)
    final_output.extend(temp_pheno_o_strs)

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
            elif k.endswith("default"):
                gen_beta_str += f"--beta {config[k]} "
            else:
                gen_beta_str += f"--{k} {config[k]} "

if config["beta_mode"] == "varmatched":
    gen_beta_h2_str = f"-h2 {config['h2']} -target_vP {config['target_vP']} "
else:
    gen_beta_h2_str = ""
print(f"gbs: {gen_beta_str}")
touch_str = "touch.file" if config["analysis_betas"] == "gwas" else "touch_grass.file"
if SLIM_FLAG:
    mem_req = max(2000,int(400000 * (num_loci / 2500) ** 2))
else:
    mem_req = int(num_loci*config['Ne']*10*25/(1000*1000))
if mem_req > 32000:
    tier_str = "tier3q"
elif mem_req > 8000:
    tier_str = "tier2q"
else:
    tier_str = "tier1q"

def generate_betas_function(wildcards, output):
    if OMEGA_GEN:
        vee_ess = wildcards.omegasomething**2
    else:
        vee_ess = wildcards.omegasomething
    return f"python generate_constants.py --beta_mode {config['beta_mode']} --freq_mode {config['freq_mode']} --vs {vee_ess} --seed {int(int(wildcards.s)+1e6*float(wildcards.omegasomething))} --scale_factor {config['scale_factor']} -n {config['num_loci']} {gen_freq_str}{gen_beta_str}{gen_beta_h2_str}-o {output}"

def run_sims_function(wildcards, input, output):
    if OMEGA_GEN:
        vee_ess = wildcards.omegasomething**2
    else:
        vee_ess = wildcards.omegasomething
    if SLIM_FLAG:
        return f'slim -s {int(int(wildcards.s) + 1e6 * float(wildcards.omegasomething))} -d beta_file="\'{input[0]}\'" -d freq_file="\'{input[1]}\'" -d vs={vee_ess} -d dz={config["dz"]} -d h2={config["h2"]} -d num_gens={config["num_gens"]} -d num_loci={config["num_loci"]} -d Ne={config["Ne"]} -d ld_output={config["ld_output"]} -d mode="\'{config["mode"]}\'" -d output_path="\'{subdir_str_dict["slims"]}{config["file_prefix"]}{wabbr}{wildcards.omegasomething}_s{wildcards.s}\'" first_slim_script.slim'
    else:
        return f'python polysim.py --seed {int(int(wildcards.s)+1e6*float(wildcards.omegasomething))} --betas_file {input[0]} --freqs_file {input[1]} --vs {vee_ess} -dz {config["dz"]} -h2 {config["h2"]} --num_gens {config["num_gens"]} --num_loci {config["num_loci"]} -n {config["Ne"]} --sel_type {config["mode"]} {big_gwas_sim_str}--output {output[0]} {output[1]} {output[2]}'

def run_sample_sims_function(wildcards, input, output):
    if config["sampling_scheme"] == "matched":
        if config["num_loci"] == "all":
            return f"python sample_sim.py -i {input} -o {output} --sampling_scheme {config["sampling_scheme"]} --sampling_table_file {true_base_dir + config["sampling_table_file"]} --betas_file {true_base_dir+config['beta_file']} --seed {int(int(wildcards.s) + 1e6 * float(wildcards.omegasomething))} --sim_source {config['sim_source']}"
        else:
            return f"python sample_sim.py -i {input} -o {output} --sampling_scheme {config["sampling_scheme"]} --sampling_table_file {true_base_dir+config["sampling_table_file"]} --betas_file {subdir_str_dict['betas']+config['file_prefix']}{wabbr}{wildcards.omegasomething}_s{wildcards.s}_betas.txt --seed {int(int(wildcards.s)+1e6*float(wildcards.omegasomething))} --sim_source {config['sim_source']}"
    else:
        return f"python sample_sim.py -i {input} -o {output} --sampling_scheme {config["sampling_scheme"]} --samples_per_timepoint {config["samples_per_timepoint"]} --num_sampling_pts {config['num_sampling_pts']} --betas_file {subdir_str_dict['betas']+config['file_prefix']}{wabbr}{wildcards.omegasomething}_s{wildcards.s}_betas.txt --seed {int(int(wildcards.s)+1e6*float(wildcards.omegasomething))} --sim_source {config['sim_source']}"

def run_analysis_function(input, output):
    if "beta_file" in config:
        return f"python analyze_fullmatched_sim.py -m {config['mode']} --vary {vary_str} {analysis_ostr} -dz {config['dz']} -h2 {config['h2']} --beta_file {true_base_dir+config['beta_file']} --global_vg {true_base_dir + config['beta_file']} {true_base_dir + config['freq_file']} --sim_source {config['sim_source']} -i {input} --output_parquet {output}"
    else:
        return f"python analyze_fullmatched_sim.py -m {config['mode']} --vary {vary_str} {analysis_ostr} -dz {config['dz']} -h2 {config['h2']} --global_vg {subdir_str_dict['betas']+config['file_prefix']}{wabbr}{min(config['omegasomething_array'])}_s0_betas.txt {subdir_str_dict['freqs']+config['file_prefix']}{wabbr}{min(config['omegasomething_array'])}_s0_freqs.txt --sim_source {config['sim_source']} -i {input} --output_parquet {output}"

def run_ll_analysis_function(input, output):
    if "beta_file" in config:
        return f"python analyze_multiple_slim_modern.py -m {config['mode']} --vary {vary_str} {analysis_ostr} --gwas -dz {config['dz']} -h2 {config['h2']} --beta_file {true_base_dir+config['beta_file']} --global_vg {true_base_dir + config['beta_file']} {true_base_dir + config['freq_file']} --sim_source {config['sim_source']} -i {input} --output_parquet {output}"
    else:
        return f"python analyze_multiple_slim_modern.py -m {config['mode']} --vary {vary_str} {analysis_ostr} --gwas -dz {config['dz']} -h2 {config['h2']} --global_vg {subdir_str_dict['betas']+config['file_prefix']}{wabbr}{min(config['omegasomething_array'])}_s0_betas.txt {subdir_str_dict['freqs']+config['file_prefix']}{wabbr}{min(config['omegasomething_array'])}_s0_freqs.txt --sim_source {config['sim_source']} -i {input} --output_parquet {output}"

def run_plot_function(input, output):
    temp_output_names = [Path(op).name for op in output]
    return f"python plot_sims.py --vary {vary_str} {bpl_string} --input_parquets {input} --labels {config['file_prefix'][:-1]} --output_names {' '.join(temp_output_names)}"

rule all:
    input:
        final_output

rule generate_betas:
    output:
        subdir_str_dict["betas"]+config["file_prefix"]+f"{wabbr}{{omegasomething}}_s{{s}}_betas.txt" if config["beta_mode"] != "matched" else true_base_dir+config['beta_file'],
        subdir_str_dict["freqs"]+config["file_prefix"]+f"{wabbr}{{omegasomething}}_s{{s}}_freqs.txt" if config["freq_mode"] != "matched" else true_base_dir+config['freq_file']
    run:
        a = generate_betas_function(wildcards, output)
        print(a)
        shell(a)

rule run_sims:
    input:
        rules.generate_betas.output[0], rules.generate_betas.output[1]
    output:
        subdir_str_dict["sims"]+config["file_prefix"]+f"{wabbr}{{omegasomething}}_s{{s}}_sim{sim_ending}",
        subdir_str_dict["sims"]+config["file_prefix"] + f"{wabbr}{{omegasomething}}_s{{s}}_{big_gwas_name_str}phenotypes{phenos_ending}" if config["ld_output"] > 0 else "",
        subdir_str_dict["sims"]+config["file_prefix"]+f"{wabbr}{{omegasomething}}_s{{s}}_{big_gwas_name_str}allgenos{genos_ending}" if config["ld_output"]>0 else ""

    resources:
        mem_mb=max(2000, mem_req),
        slurm_partition=tier_str
    run:
        a = run_sims_function(wildcards, input, output)
        print(a)
        shell(a)

if config["analysis_betas"] == "gwas":
    rule perform_gwasim:
        input:
            rules.run_sims.output[2], rules.run_sims.output[1]
        output:
            subdir_str_dict["betas"]+config["file_prefix"]+f"{wabbr}{{omegasomething}}_s{{s}}_betas_{big_gwas_name_str}gwas.txt"
        resources:
            mem_mb=7500
        shell:
            f"python perform_gwasim.py -p {{input[1]}} -g {{input[0]}} -o {{output}} --source {config['sim_source']}"

    rule touch_gwas:
        input:
            expand(subdir_str_dict["betas"]+config["file_prefix"]+f"{wabbr}{{omegasomething}}_s{{s}}_betas_{big_gwas_name_str}gwas.txt", omegasomething=config["omegasomething_array"], s=range(config["num_replicates"]))
        output:
            touch(true_base_dir+"touch.file")
else:
    rule dont_perform_gwasim:
        output:
            touch(true_base_dir+"touch_grass.file")

rule sample_data:
    input:
        rules.run_sims.output[0]
    output:
        subdir_str_dict["data"]+config["file_prefix"]+f"{wabbr}{{omegasomething}}_s{{s}}_data.csv",
        subdir_str_dict["trajs"]+config['file_prefix']+f"{wabbr}{{omegasomething}}_s{{s}}_trajs.pdf"
    run:
        a = run_sample_sims_function(wildcards,input,output)
        print(a)
        shell(a)

rule run_grids:
    input:
        rules.sample_data.output[0]
    output:
        subdir_str_dict[grids_str]+config['file_prefix']+f"{wabbr}{{omegasomething}}_s{{s}}_grid.csv",
        subdir_str_dict[grids_str]+config['file_prefix']+f"{wabbr}{{omegasomething}}_s{{s}}_grid_uncon.csv"
    resources:
        threads=8,
        mem_mb = 5000
    shell:
        f"python run_hmm.py {{input}} {{output[0]}} --time_after_zero -hs {config['hidden_states']} -sid {config['hmm_init_dist']}"
        f" {cond_str}--grid_s_max {config['grid_s_max']} -np {config['num_half_grid_points']} -Ne {config['hmm_Ne']} --progressbar --snakemake -nc {{resources.threads}}"

rule analyze_w_grids_uncon:
    input:
        expand(subdir_str_dict[grids_str]+config['file_prefix']+f"{wabbr}{{omegasomething}}_s{{s}}_grid_uncon.csv", omegasomething=config["omegasomething_array"], s=range(config["num_replicates"])),
        true_base_dir+touch_str
    output:
        true_base_dir + config["file_prefix"] + "regression.parquet"
    resources:
        mem_mb=2000
    run:
        a = run_analysis_function(input, output)
        print(a)
        shell(a)

# rule analyze_w_grids_ll:
#     input:
#         expand(subdir_str_dict[grids_str]+config['file_prefix']+f"{wabbr}{{omegasomething}}_s{{s}}_grid_uncon.csv", omegasomething=config["omegasomething_array"], s=range(config["num_replicates"])),
#         true_base_dir+touch_str
#     output:
#         true_base_dir+config['file_prefix']+"ll.parquet"
#     resources:
#         mem_mb=2000
#     run:
#         a = run_ll_analysis_function(input, output)
#         print(a)
#         shell(a)

rule plot_grids:
    input:
        rules.analyze_w_grids_uncon.output[0]
    output:
        output_plots
    run:
        a = run_plot_function(input, output)
        print(a)
        shell(a)

# rule plot_grids_ll:
#     input:
#         rules.analyze_w_grids_ll.output[0]
#     output:
#         output_plots_ll
#     run:
#         a = run_plot_function(input, output)
#         print(a)
#         shell(a)

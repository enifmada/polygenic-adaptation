import json
from pathlib import Path

base_dir = "../../../polyoutput/slim_testing/"+config["file_prefix"][:-1]
Path(base_dir).mkdir(parents=True, exist_ok=True)

true_base_dir = base_dir + "/"
subdir_strs = ["betas", "freqs", "slims", "data", "trajs"]
subdir_singles = ["betas", "freqs", "slim", "data", "trajs", "grid", "surface"]
subdir_ftypes = [".txt", ".txt", ".txt", ".csv", ".pdf", ".csv", ".pdf"]

if "end_gen" in config and "end_ns" in config:
    cond_str = f"--condition_on_seg --end_gen {config['end_gen']} --end_ns {config['end_ns']} "
    cond_short_str = "cond_"
    COND_FLAG = True
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

possible_vars = ["betascale", "loci", "h2"]

assert config["vary_var"] in possible_vars
for var in possible_vars:
    if var == config["vary_var"]:
        config[f"{var}_array"] = config["vary_array"]

config["omega_array"] = config["omega"] * len(config["vary_array"])
for s_i, subdir in enumerate(subdirs):
    subdir_str_dict[subdir_strs[s_i]] = subdir
    Path(subdir[:-1]).mkdir(parents=True, exist_ok=True)
    temp_strs = expand(subdir+config['file_prefix']+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_"+subdir_singles[s_i]+subdir_ftypes[s_i], v=config["vary_array"], s=range(config["num_replicates"]))
    if "surface" not in subdir:
        final_output.extend(temp_strs)

final_output.append(true_base_dir+config['file_prefix']+f"{config["vary_var"][0]}_{cond_short_str}lineplot_truebetas_regression.pdf")
final_output.append(true_base_dir+config['file_prefix']+f"{config["vary_var"][0]}_{cond_short_str}lineplot_gwasbetas_regression.pdf")
final_output.append(true_base_dir+config['file_prefix']+f"{config["vary_var"][0]}_{cond_short_str}lineplot_regression_Serrplot.pdf")
final_output.append(true_base_dir+config['file_prefix']+f"{config["vary_var"][0]}_{cond_short_str}lineplot_regression_werrplot.pdf")

if "hmm_Ne" not in config:
    config["hmm_Ne"] = config["Ne"]

if "ld_output" not in config:
    config["ld_output"] = -1

if "init_freq" in config:
    config["freq_init"] = config["init_freq"]

if "analysis_betas" not in config:
    config["analysis_betas"] = "ground_truth"

if config["analysis_betas"] == "gwas":
    #need the ld output to do a gwas
    if config["ld_output"] < 0:
        config["ld_output"] = 5
    temp_betas_w_strs = expand(subdir_str_dict["betas"]+config['file_prefix']+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_betas_gwas.txt", v=config["vary_array"], s=range(config["num_replicates"]))
    final_output.extend(temp_betas_w_strs)

if config["ld_output"] > 0:
    temp_vcf_w_strs = expand(subdir_str_dict["slims"]+config['file_prefix']+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_allgenos.vcf", v=config["vary_array"], s=range(config["num_replicates"]))
    final_output.extend(temp_vcf_w_strs)
    temp_pheno_w_strs = expand(subdir_str_dict["slims"]+config['file_prefix']+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_phenotypes.txt", v=config["vary_array"], s=range(config["num_replicates"]))
    final_output.extend(temp_pheno_w_strs)

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


#this probably does not work
def generate_betas_function(wildcards, output):
    if config['vary_var'] == 'loci':
        return f"python generate_constants.py --beta_mode {config['beta_mode']} --freq_mode {config['freq_mode']} --omega {config["omega"]} --seed {int(int(wildcards.s)+1000*float(wildcards.v))} -n {wildcards.v} {gen_freq_str}{gen_beta_str}-o {output}"
    elif config['vary_var'] == "betascale":
        return f"python generate_constants.py --beta_mode {config['beta_mode']} --freq_mode {config['freq_mode']} --omega {config["omega"]} --seed {int(int(wildcards.s) + 1000 * float(wildcards.v))} -n {config['num_loci']} --scale_factor {wildcards.v} {gen_freq_str}{gen_beta_str}-o {output}"
    else:
        return f"python generate_constants.py --beta_mode {config['beta_mode']} --freq_mode {config['freq_mode']} --omega {config["omega"]} --seed {int(int(wildcards.s)+1000*float(wildcards.v))} -n {config['num_loci']} {gen_freq_str}{gen_beta_str}-o {output}"

def generate_slims_function(wildcards, input):
    if config['vary_var'] == 'loci':
        return f'slim -s {int(int(wildcards.s)+1000*float(wildcards.v))} -d beta_file="\'{input[0]}\'" -d freq_file="\'{input[1]}\'" -d omega={config["omega"]} -d dz={config["dz"]} -d h2={config["h2"]} -d num_loci={wildcards.v} -d num_gens={config["num_gens"]} -d Ne={config["Ne"]} -d ld_output={config["ld_output"]} -d mode="\'{config["mode"]}\'" -d output_path="\'{subdir_str_dict["slims"]}{config["file_prefix"]}w{config["omega"]}_s{wildcards.s}_{config["vary_var"][0]}{wildcards.v}\'" first_slim_script.slim'
    elif config['vary_var'] == 'h2':
        return f'slim -s {int(int(wildcards.s)+1000*float(wildcards.v))} -d beta_file="\'{input[0]}\'" -d freq_file="\'{input[1]}\'" -d omega={config["omega"]} -d dz={config["dz"]} -d h2={wildcards.v} -d num_loci={config["num_loci"]} -d num_gens={config["num_gens"]} -d Ne={config["Ne"]} -d ld_output={config["ld_output"]} -d mode="\'{config["mode"]}\'" -d output_path="\'{subdir_str_dict["slims"]}{config["file_prefix"]}w{config["omega"]}_s{wildcards.s}_{config["vary_var"][0]}{wildcards.v}\'" first_slim_script.slim'
    else:
        return f'slim -s {int(int(wildcards.s)+1000*float(wildcards.v))} -d beta_file="\'{input[0]}\'" -d freq_file="\'{input[1]}\'" -d omega={config["omega"]} -d dz={config["dz"]} -d h2={config["h2"]} -d num_loci={config["num_loci"]} -d num_gens={config["num_gens"]} -d Ne={config["Ne"]} -d ld_output={config["ld_output"]} -d mode="\'{config["mode"]}\'" -d output_path="\'{subdir_str_dict["slims"]}{config["file_prefix"]}w{config["omega"]}_s{wildcards.s}_{config["vary_var"][0]}{wildcards.v}\'" first_slim_script.slim'

def generate_analysis_function(input, output):
    if config['vary_var'] == 'h2':
        return f'python analyze_multiple_slim_regression.py -m {config['mode']} {'--gwas ' if config['analysis_betas'] == 'gwas' else ''}--vary {config['vary_var']} -dz {config['dz']} --regmode weighted --global_vg {true_beta_path} {true_freq_path} -i {input} -o {output}'
    else:
        return f'python analyze_multiple_slim_regression.py -m {config['mode']} {'--gwas ' if config['analysis_betas'] == 'gwas' else ''}--vary {config['vary_var']} -dz {config['dz']} --regmode weighted -h2 {config['h2']} --global_vg {true_beta_path} {true_freq_path} -i {input} -o {output}'

rule all:
    input:
        final_output

rule generate_betas:
    output:
        subdir_str_dict["betas"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_betas.txt",
        subdir_str_dict["freqs"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_freqs.txt"
    run:
        a = generate_betas_function(wildcards, output)
        print(a)
        shell(a)

if config["analysis_betas"] == "gwas":
    rule perform_gwaslim:
        input:
            subdir_str_dict["slims"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_allgenos.vcf",
            subdir_str_dict["slims"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_phenotypes.txt"
        output:
            subdir_str_dict["betas"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_betas_gwas.txt"
        resources:
            mem_mb=10000
        shell:
            f"python perform_gwaslim.py -p {{input[1]}} -g {{input[0]}} -o {{output}}"

    rule touch_gwas:
        input:
            expand(subdir_str_dict["betas"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_betas_gwas.txt", v=config["vary_array"], s=range(config["num_replicates"]))
        output:
            touch(true_base_dir+"touch.file")
else:
    rule dont_perform_gwaslim:
        output:
            touch(true_base_dir+"touch_grass.file")

rule sample_data:
    input:
        subdir_str_dict["slims"]+config['file_prefix']+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_slim.txt"
    output:
        subdir_str_dict["data"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_data.csv",
        subdir_str_dict["trajs"]+config['file_prefix']+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_trajs.pdf"
    shell:
        f"python sample_slim.py -i {{input}} -o {{output}} --sampling_scheme fixed --samples_per_timepoint {config['samples_per_timepoint']}"
        f" --num_sampling_pts {config['num_sampling_pts']} --beta_file {{rules.generate_betas.output[0]}}"#output

rule run_slims:
    input:
        subdir_str_dict["betas"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_betas.txt",
        subdir_str_dict["freqs"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_freqs.txt"
    output:
        subdir_str_dict["slims"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_slim.txt",
        subdir_str_dict["slims"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_allgenos.vcf" if config["ld_output"]>0 else "",
        subdir_str_dict["slims"]+config["file_prefix"]+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_phenotypes.txt" if config["ld_output"]>0 else ""
    resources:
        mem_mb=50000
    run:
        a = generate_slims_function(wildcards, input)
        print(a)
        shell(a)

rule run_grids:
    input:
        subdir_str_dict["data"]+config['file_prefix']+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_data.csv"
    output:
        subdir_str_dict[grids_str]+config['file_prefix']+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_grid.csv"
    resources:
        threads=3,
        #mem_mb = 5000
    shell:
        f"python run_hmm.py {{input}} {{output}} --time_after_zero -hs {config['hidden_states']} -sid {config['hmm_init_dist']}"
        f" {cond_str}--grid_s_max 0.25 -np {config['num_half_grid_points']} -Ne {config['hmm_Ne']} --progressbar --snakemake -nc {{resources.threads}}"

rule analyze_w_grids_regression:
    input:
        expand(subdir_str_dict[grids_str]+config['file_prefix']+f"w{config['omega']}_s{{s}}_{config['vary_var'][0]}{{v}}_grid.csv", v=config["vary_array"], s=range(config["num_replicates"])),
        true_base_dir+touch_str
    output:
        true_base_dir+config['file_prefix']+f"{config["vary_var"][0]}_{cond_short_str}lineplot_truebetas_regression.pdf",
        true_base_dir+config['file_prefix']+f"{config["vary_var"][0]}_{cond_short_str}lineplot_gwasbetas_regression.pdf",
        true_base_dir+config['file_prefix']+f"{config["vary_var"][0]}_{cond_short_str}lineplot_regression_Serrplot.pdf",
        true_base_dir+config['file_prefix']+f"{config["vary_var"][0]}_{cond_short_str}lineplot_regression_werrplot.pdf"
    resources:
        mem_mb=5000
    run:
        a = generate_analysis_function(input, output)
        print(a)
        shell(a)#lol

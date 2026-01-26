from pathlib import Path

base_dir = Path("../../../polyoutput/ukbb_shrunk")
#scratch_dir = Path("../../../polyoutput/ukbb_shrunk")
scratch_dir = Path("../../../../../../../../scratch/afine2")

adna_snps_info_file = base_dir/config['adna_info_file']
adna_full_csv_file = base_dir/config['adna_data_file']

pheno_descr_file = base_dir/config['pheno_descr_file']


base_dir.mkdir(exist_ok=True)
Path(base_dir/"output").mkdir(exist_ok=True)
#scratch_dir.mkdir(parents=True, exist_ok=True)

if "end_gen" in config and "end_ns" in config:
    cond_str = f"--condition_on_seg --end_gen {config['end_gen']} --end_ns {config['end_ns']} "
    cond_short_str = "cond_"
    COND_FLAG = True
else:
    cond_str = ""
    cond_short_str = ""
    COND_FLAG = False

scratch_subdir_strs = ["LD", "LD_subset", "phenos", "phenos_shrunk", "phenos_subset", f"clumped_data{config['suffix']}"]
scratch_subdir_strs = ["UKBB_"+s for s in scratch_subdir_strs]

if COND_FLAG:
    scratch_subdir_strs.extend([f"UKBB_grids{config['suffix']}cond"])
    grids_str = "grids_cond"
else:
    scratch_subdir_strs.extend([f"UKBB_grids{config['suffix']}"])
    grids_str = "grids"
for subdir in scratch_subdir_strs:
    Path(scratch_dir/subdir).mkdir(exist_ok=True)

with open(base_dir/config['pheno_list_file'], "r") as f:
    phenos = f.readlines()

phenos = [p.strip() for p in phenos]
#phenos = [p for p in phenos if "2966_irnt" in p]
pheno_codes = [p.partition(".")[0] for p in phenos]
phenos_simple = [p + ".tsv.bgz" for p in pheno_codes]

with open(base_dir/config['ld_list_file'], "r") as f:
    ld_files = f.readlines()

ld_files = [l.strip() for l in ld_files]
ld_basestrs = [l.partition("/")[-1] for l in ld_files if ".gz" not in l]
ld_superbasestrs = [l.rpartition(".")[0] for l in ld_basestrs]

clump_params = ["min_r2", "min_height", "window_size_kb"]
clump_defaults = [0.5, 1e-3, 250]


if "hmm_Ne" not in config:
    config["hmm_Ne"] = 9715

for c_i, clump_param in enumerate(clump_params):
    if clump_param not in config:
        config[clump_param] = clump_defaults[c_i]

rule all:
    input:
        #expand(base_dir/"ash_results/{trait}_ash.csv.gz", trait=TRAITS), expand(base_dir/"ash_results/{trait}_ash_g.csv", trait=TRAITS), expand(base_dir/"ash_results/{trait}_density.pdf", trait=TRAITS), base_dir/f"output/all_gvars{config['suffix']}.txt",
        base_dir/f"output/all_dir_Sdz_ests{config['suffix']}_presfinal.pdf", base_dir/f"output/all_dir_z_ests{config['suffix']}_presfinal.pdf",
        base_dir/f"output/all_stab_S_ests{config['suffix']}_presfinal.pdf", base_dir/f"output/all_stab_z_ests{config['suffix']}_presfinal.pdf",
        base_dir/f"output/all_stab_w_ests{config['suffix']}_presfinal.pdf", base_dir/f"output/all_analysis_data{config['suffix']}_{cond_short_str}final.parquet"

rule download_ld:
    output:
        expand(scratch_dir/f"{scratch_subdir_strs[0]}/{{ldstr}}", ldstr=ld_basestrs), expand(scratch_dir/f"{scratch_subdir_strs[0]}/{{ldstr}}.gz", ldstr=ld_superbasestrs)
    resources:
        time="0-20:00:00"
    shell:
        f"python download_ld.py --output_dir {scratch_dir}"

rule subset_ld:
    input:
        rules.download_ld.output
        #expand(scratch_dir / f"{scratch_subdir_strs[0]}/{{ldstr}}",ldstr=ld_basestrs),expand(scratch_dir / f"{scratch_subdir_strs[0]}/{{ldstr}}.gz",ldstr=ld_superbasestrs)
    output:
        expand(scratch_dir/f"{scratch_subdir_strs[1]}/{{ldstr}}_subset.npz", ldstr=ld_superbasestrs), expand(scratch_dir/f"{scratch_subdir_strs[1]}/{{ldstr}}_rsids.txt", ldstr=ld_superbasestrs)
    resources:
        time="0-15:00:00",
        threads=1,
        mem_mb = 32000,
        slurm_partition = "tier2q"
    shell:
        f"python subset_ld.py --output_dir {scratch_dir/scratch_subdir_strs[1]} --ld_file_list {base_dir/config['ld_list_file']} --adna_data_file {adna_snps_info_file} -nc {{resources.threads}}"

rule download_phenos:
    output:
        expand(scratch_dir/f"{scratch_subdir_strs[2]}/{{phenostr}}", phenostr=phenos_simple)
    resources:
        time="0-02:00:00"
    shell:
        f"python download_phenos.py --pheno_list_file {base_dir/config['pheno_list_file']} --pheno_loc_file {base_dir/config['pheno_loc_file']} --output_dir {scratch_dir/scratch_subdir_strs[2]}"

rule shrink_phenos:
    input:
        scratch_dir/f"{scratch_subdir_strs[2]}/{{phenostr}}.tsv.bgz"
    output:
        scratch_dir/f"{scratch_subdir_strs[3]}/{{phenostr}}_ash.parquet", scratch_dir/f"{scratch_subdir_strs[3]}/{{phenostr}}_ash_g.csv"
    resources:
        mem_mb = 70000,
        slurm_partition = "tier3q"
    shell:
        "python shrink_ukbb_sumstats.py -i {input} -o {output}"

rule subset_phenos:
    input:
        rules.shrink_phenos.output[0]#scratch_dir/f"{scratch_subdir_strs[3]}/{{phenostr}}_ash.parquet"
    output:
        scratch_dir/f"{scratch_subdir_strs[4]}/{{phenostr}}_subset.parquet"
    resources:
        time="0-00:10:00",
        mem_mb = 20000
    shell:
        f"python subset_phenos.py --input {{input}} --output {{output}} --adna_data_file {adna_snps_info_file}"
#
rule clump_data:
    input:
        scratch_dir/f"{scratch_subdir_strs[4]}/{{phenostr}}_subset.parquet",
        expand(scratch_dir / f"{scratch_subdir_strs[1]}/{{ldstr}}_subset.npz", ldstr=ld_superbasestrs),
        expand(scratch_dir / f"{scratch_subdir_strs[1]}/{{ldstr}}_rsids.txt",ldstr=ld_superbasestrs)
        #rules.subset_ld.output
    output:
        scratch_dir/f"{scratch_subdir_strs[5]}/{{phenostr}}_clumped{config['suffix']}.parquet"
    resources:
        time="0-00:20:00",
        mem_mb = 5000
    shell:
        f"python clump_effects.py --ld_dir {scratch_dir/scratch_subdir_strs[1]} --pheno_file {{input[0]}}"
        f" --clump_dir {scratch_dir/scratch_subdir_strs[5]} --ld_list_file {base_dir/config['ld_list_file']}"
        f" --clump_suffix _clumped{config['suffix']}.parquet --min_r2 {config['min_r2']}"
        f" --min_height {config['min_height']} --window_size_kb {config['window_size_kb']}"

rule run_grid:
    input:
        adna_full_csv_file, adna_snps_info_file, scratch_dir/f"{scratch_subdir_strs[5]}/{{phenostr}}_clumped{config['suffix']}.parquet"
    output:
        scratch_dir/f"{scratch_subdir_strs[6]}/{{phenostr}}_grid{config['suffix']}{cond_short_str[:-1]}_uncon.csv"
    resources:
        threads=8,
        mem_mb=20000,
        time="0-03:00:00",
        slurm_partition="tier1q"
    shell:
        f"python run_hmm.py {{input[0]}} {{output}} --time_after_zero -hs {config['hidden_states']} -sid {config['hmm_init_dist']}"
        f" -nc {{resources.threads}} --grid_s_max {config['grid_s_max']} -np {config['num_half_grid_points']} -Ne {config['hmm_Ne']}"
        f" {cond_str}--subset_input {{input[1]}} {{input[2]}} --progressbar --snakemake"

rule analyze_grids:
    input:
        expand(scratch_dir/f"{scratch_subdir_strs[6]}/{{phenostr}}_grid{config['suffix']}{cond_short_str[:-1]}_uncon.csv", phenostr=pheno_codes),  expand(scratch_dir/f"{scratch_subdir_strs[5]}/{{phenostr}}_clumped{config['suffix']}.parquet", phenostr=pheno_codes)
    output:
        base_dir/f"output/all_analysis_data{config['suffix']}_{cond_short_str}final.parquet"
    resources:
        mem_mb=50000,
        threads=1,
        time="0-02:00:00"
    shell:
        f"python analyze_real_data.py --max_jk_blocks 40 --pheno_descr_file {pheno_descr_file} -nc {{resources.threads}} -i {{input}} --output_parquet {{output}}"

rule plot_data:
    input:
        base_dir / f"output/all_analysis_data{config['suffix']}_{cond_short_str}final.parquet"
    output:
        rules.all.input[:-1]
    resources:
        time="0-00:10:00"
    shell:
        "python plot_real_data.py --parquet_file {input} -o {output} --no_below_zero"

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import seaborn as sns

data_file = "../../../data/UKBB_GWAS/blood_EOSINOPHIL_COUNT.sumstats.gz"


data_array = pd.read_csv(data_file, sep="\t")


big_betas = data_array.loc[data_array["Beta"] > .05, ["Beta", "se"]]
ploot = sns.relplot(x=big_betas["Beta"], y=big_betas["se"], data=big_betas)
ploot.figure.savefig("test_plot.png", format="png", bbox_inches="tight")




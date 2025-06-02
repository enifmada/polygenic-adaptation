from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd

data_file = "../../../data/UKBB_GWAS/blood_EOSINOPHIL_COUNT.sumstats.gz"


data_array = pd.read_csv(data_file, sep="\t")
# raise Error
# data_array.to_csv("blood_EOSINOPHIL_COUNT_ashvalues.csv.gz", columns=["Beta", "se"], compression="gzip")

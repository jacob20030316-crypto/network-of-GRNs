"""Reading a TCGA project's clinical matrix.

TCGA does not ship a series matrix. Its clinical variables live in a separate
file beside the expression data, and the directory a disease lives in is named
by the study's own abbreviation rather than by the disease label this project
uses, so finding it is a lookup rather than a path join.

This is what remains of a larger module that also decided which of those
variables could be used. That decision is now the axis agent's.
"""
import glob
import os
import re

import pandas as pd

NULLS = {"[Not Available]", "[Unknown]", "[Not Applicable]", "[Discrepancy]",
         "[Not Evaluated]", "", "NA", "nan"}


def clinical_matrix(tcga_dir):
    f = glob.glob(os.path.join(tcga_dir, "*clinicalMatrix"))
    return pd.read_csv(f[0], sep="\t", index_col=0, low_memory=False) if f else None


def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower().replace("_and_", "_"))


def tcga_dir_for(cohort_dirs, disease):
    """Match our disease name onto a TCGA_<name>_(<CODE>) directory."""
    if disease in DIR_ALIASES:
        for d in cohort_dirs:
            if os.path.basename(d) == DIR_ALIASES[disease]:
                return d
    key = _norm(disease)
    for d in cohort_dirs:
        name = re.sub(r"^TCGA_|\(.*\)$", "", os.path.basename(d)).strip("_")
        if _norm(name) == key:
            return d
    return None

"""What a cohort recorded about its patients, as the submitters wrote it.

This reads and counts; it does not judge. Which of these variables can divide
patients into comparable groups is a question about what the variable means,
and it is answered by the axis agent, which reads what this returns.

Two repositories, two shapes. A GEO series matrix carries a handful of
`!Sample_characteristics_ch1` rows, each a key and one value per sample, and
what the key means is whatever the submitter decided. TCGA carries a clinical
matrix of several hundred columns with stable names, most of them empty for any
given project.

Both are reduced to the same thing: a variable, its levels, and how many
samples fall in each.
"""
import glob
import gzip
import os
import re
from collections import Counter

import pandas as pd

from grn import paths as config

MAX_LEVELS = 12          # beyond this a variable is free text, not a partition
PREVIEW_LEVELS = 8       # how many levels to show the agent per variable


def _split(line):
    return [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]


def geo_variables(disease, cohort, samples=None):
    """The characteristics rows of one GEO series, restricted to given samples."""
    files = sorted(glob.glob(
        os.path.join(config.RAW_GEO, disease, cohort, "*series_matrix.txt.gz")))
    if not files:
        return None
    ids, rows = None, []
    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("!Sample_geo_accession"):
                    ids = _split(line)[1:]
                elif line.startswith("!Sample_characteristics_ch1"):
                    rows.append(_split(line)[1:])
                elif line.startswith("!series_matrix_table_begin"):
                    break
    if ids is None:
        return None

    keep = [i for i, s in enumerate(ids)
            if samples is None or s in set(samples)]
    out = []
    for vals in rows:
        vals = [vals[i] if i < len(vals) else "" for i in keep]
        # A characteristics row is usually "key: value" with one key repeated,
        # but a submitter may use the row for several keys at once.
        keys, cleaned = [], []
        for v in vals:
            if ":" in v:
                k, _, rest = v.partition(":")
                keys.append(k.strip().lower())
                cleaned.append(rest.strip())
            else:
                keys.append("")
                cleaned.append(v.strip())
        key = Counter(k for k in keys if k).most_common(1)
        key = key[0][0] if key else "(unlabelled)"
        counts = Counter(v for v in cleaned if v not in ("", "NA", "na", "--"))
        if counts:
            out.append({"variable": key, "n_levels": len(counts),
                        "levels": counts.most_common(PREVIEW_LEVELS),
                        "n_annotated": sum(counts.values())})
    return {"disease": disease, "cohort": cohort, "source": "GEO",
            "n_samples": len(keep), "variables": out}


def tcga_variables(disease, samples=None):
    """The clinical columns of one TCGA project, restricted to given samples."""
    from grn.tools.curation.tcga import clinical_matrix, tcga_dir_for
    dirs = sorted(glob.glob(os.path.join(config.RAW_TCGA, "*")))
    d = tcga_dir_for(dirs, disease)
    if d is None:
        return None
    C = clinical_matrix(d)
    if C is None or not len(C):
        return None
    if samples is not None:
        C = C[C.index.astype(str).isin(set(samples))]
    out = []
    for col in C.columns:
        v = C[col].astype(str).str.strip()
        v = v[~v.isin(["", "nan", "[Not Available]", "[Unknown]",
                       "[Not Applicable]", "[Discrepancy]",
                       "[Not Evaluated]", "[Completed]"])]
        if not len(v):
            continue
        counts = Counter(v)
        if len(counts) > 60:                 # identifiers, dates, free text
            continue
        out.append({"variable": col, "n_levels": len(counts),
                    "levels": counts.most_common(PREVIEW_LEVELS),
                    "n_annotated": int(len(v))})
    return {"disease": disease, "cohort": "TCGA", "source": "TCGA",
            "n_samples": int(len(C)), "variables": out}


def variables_for(disease, cohort, samples=None):
    """Whichever repository this cohort came from."""
    if str(cohort).upper() == "TCGA":
        return tcga_variables(disease, samples)
    return geo_variables(disease, cohort, samples)


def describe(packet, max_variables=60):
    """The packet as the agent reads it."""
    lines = [f"disease: {packet['disease']}",
             f"cohort: {packet['cohort']}  ({packet['source']})",
             f"patients in the disease group: {packet['n_samples']}",
             "",
             "Counts below are over every sample in the cohort, including any "
             "that are not patients of this disease. The disease group is "
             "taken first and the axes are applied inside it, so treat the "
             "counts as indicative of relative size rather than as the size of "
             "the groups that will result.",
             "", "variables recorded, with how many samples fall in each level:"]
    for v in packet["variables"][:max_variables]:
        shown = "; ".join(f"{lv} = {n}" for lv, n in v["levels"])
        more = "" if v["n_levels"] <= len(v["levels"]) else \
            f"  (+{v['n_levels'] - len(v['levels'])} more levels)"
        lines.append(f"  {v['variable']}")
        lines.append(f"      {shown}{more}")
    if len(packet["variables"]) > max_variables:
        lines.append(f"  (+{len(packet['variables']) - max_variables} more variables)")
    return "\n".join(lines)

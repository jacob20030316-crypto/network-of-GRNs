"""Turn the axis decisions into the patient groups the networks are built on.

The axis agent says which recorded variable identifies patients of the disease,
and which variables divide them, naming for each level the raw values that
belong to it. This applies those definitions to the samples and writes the
registry every later stage reads.

Nothing is decided here. The split is arithmetic on rules that were written
down, which is what makes a patient group checkable: the registry records the
axis, the level and the sample ids, and the rule that produced them is in
`condition_axes.csv` beside it.

Two networks come out:

  uncond   one group per cohort, all its patients pooled. The floor is higher
           because this group has to stand on its own.
  cond     one group per (cohort, axis, level). A group here is one view of one
           disease, and two diseases are compared through every pairing of
           their views, so the floor is lower.

A level that does not reach its floor is dropped rather than merged. Merging
two levels of a clinical variable makes a group that does not correspond to
anything a clinician would recognise, and the comparison it enters would then
be against a group that does.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

from grn import paths as config
from grn.tools.curation import characteristics as ch

FLOOR_UNCOND = 50
FLOOR_COND = 30


def _bounds(expr):
    """Parse a numeric band as the agent writes it: '>=40 and <60', '<80'."""
    out = []
    for m in re.finditer(r"(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)", str(expr)):
        out.append((m.group(1), float(m.group(2))))
    return out


def _in_band(value, bands):
    try:
        x = float(str(value).strip().split()[0])
    except (ValueError, IndexError):
        return False
    for op, cut in bands:
        if op == ">=" and not x >= cut:
            return False
        if op == ">" and not x > cut:
            return False
        if op == "<=" and not x <= cut:
            return False
        if op == "<" and not x < cut:
            return False
    return bool(bands)


def _values_by_sample(disease, cohort, variable):
    """sample id -> the raw value of one recorded variable."""
    if str(cohort).upper() == "TCGA":
        from grn.tools.curation.tcga import clinical_matrix, tcga_dir_for
        import glob
        d = tcga_dir_for(sorted(glob.glob(os.path.join(config.RAW_TCGA, "*"))),
                         disease)
        if d is None:
            return {}
        C = clinical_matrix(d)
        if C is None or variable not in C.columns:
            return {}
        return {str(k): str(v).strip() for k, v in C[variable].items()}

    import glob
    import gzip
    files = sorted(glob.glob(os.path.join(config.RAW_GEO, disease, cohort,
                                          "*series_matrix.txt.gz")))
    ids, rows = None, []
    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("!Sample_geo_accession"):
                    ids = ch._split(line)[1:]
                elif line.startswith("!Sample_characteristics_ch1"):
                    rows.append(ch._split(line)[1:])
                elif line.startswith("!series_matrix_table_begin"):
                    break
    if ids is None:
        return {}
    for vals in rows:
        keys = [v.partition(":")[0].strip().lower() if ":" in v else ""
                for v in vals]
        key = pd.Series([k for k in keys if k]).mode()
        if not len(key) or key.iloc[0] != str(variable).lower():
            continue
        out = {}
        for s, v in zip(ids, vals):
            out[str(s)] = v.partition(":")[2].strip() if ":" in v else v.strip()
        return out
    return {}


def _matrix_samples(disease, cohort):
    """The samples that survived preprocessing, in the order the matrix holds."""
    path = config.matrix_path(disease, cohort)
    if not os.path.isfile(path):
        return []
    return [str(s) for s in pd.read_csv(path, usecols=[0], index_col=0).index]


def build(axes_path=None, floor_uncond=FLOOR_UNCOND, floor_cond=FLOOR_COND):
    A = pd.read_csv(axes_path or config.out("condition_axes.csv"))
    A["values"] = A["values"].fillna("")
    rows, dropped = [], []

    for (disease, cohort), grp in A.groupby(["disease", "cohort"], sort=True):
        have = _matrix_samples(disease, cohort)
        if not have:
            dropped.append((disease, cohort, "", "no expression matrix", 0))
            continue
        source = "TCGA" if str(cohort).upper() == "TCGA" else "GEO"

        # the disease group first: the axes divide it, not the whole cohort
        cases = set(have)
        dg = grp[grp.axis == "disease_group"]
        if len(dg):
            r = dg.iloc[0]
            vals = {v for v in str(r["values"]).split("|") if v}
            by = _values_by_sample(disease, cohort, r.variable)
            if by:
                cases = {s for s in have if by.get(s) in vals}
            else:
                dropped.append((disease, cohort, "disease_group",
                                f"variable {r.variable!r} not readable", 0))
        if len(cases) < floor_cond:
            dropped.append((disease, cohort, "", "disease group below floor",
                            len(cases)))
            continue

        rows.append(dict(unit_id=f"{disease}||{cohort}||disease_group=all",
                         disease=disease, cohort=cohort, source=source,
                         axis="disease_group", level="all", n=len(cases),
                         network="uncond" if len(cases) >= floor_uncond else "",
                         samples="|".join(sorted(cases))))

        for axis, ax in grp[grp.axis != "disease_group"].groupby("axis"):
            by = _values_by_sample(disease, cohort, ax.iloc[0].variable)
            if not by:
                dropped.append((disease, cohort, axis,
                                f"variable {ax.iloc[0].variable!r} not readable", 0))
                continue
            for _, lv in ax.iterrows():
                if str(lv.kind) == "numeric_binned":
                    bands = _bounds(lv.bounds)
                    members = {s for s in cases if _in_band(by.get(s, ""), bands)}
                else:
                    vals = {v for v in str(lv["values"]).split("|") if v}
                    members = {s for s in cases if by.get(s) in vals}
                if len(members) < floor_cond:
                    dropped.append((disease, cohort, f"{axis}={lv.level}",
                                    "below floor", len(members)))
                    continue
                rows.append(dict(
                    unit_id=f"{disease}||{cohort}||{axis}={lv.level}",
                    disease=disease, cohort=cohort, source=source,
                    axis=axis, level=lv.level, n=len(members), network="cond",
                    samples="|".join(sorted(members))))

    U = pd.DataFrame(rows)
    U = U[U.network != ""].reset_index(drop=True)
    D = pd.DataFrame(dropped, columns=["disease", "cohort", "axis",
                                       "reason", "n"])
    return U, D


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--axes", default=None)
    ap.add_argument("--floor-uncond", type=int, default=FLOOR_UNCOND)
    ap.add_argument("--floor-cond", type=int, default=FLOOR_COND)
    args = ap.parse_args(argv)

    config.require_raw_geo()
    U, D = build(args.axes, args.floor_uncond, args.floor_cond)
    if not len(U):
        raise SystemExit("no patient groups were formed; check condition_axes.csv")

    config.write_table(U, config.out("analysis_units.csv"), min_rows=50)
    config.write_table(D, config.out("analysis_units_dropped.csv"), min_rows=0)

    u, c = (U.network == "uncond").sum(), (U.network == "cond").sum()
    print(f"\n  {len(U)} patient groups: {u} pooled (floor {args.floor_uncond}), "
          f"{c} divided (floor {args.floor_cond})")
    print(f"  {U[U.network == 'cond'].axis.nunique()} clinical axes over "
          f"{U.groupby(['disease', 'cohort']).ngroups} cohorts, "
          f"{U.disease.nunique()} diseases")
    print(f"  median group size {int(U.n.median())}")
    if len(D):
        print(f"\n  {len(D)} candidate groups not formed:")
        print(D.reason.value_counts().to_string())
    return 0


def run(**kw):
    argv = []
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]
    return main(argv)


if __name__ == "__main__":
    sys.exit(main())

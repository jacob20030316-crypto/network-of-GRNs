"""Assemble one evidence packet per candidate cohort, for independent adjudication.

The packet deliberately separates FIRST-HAND evidence (raw GEO sample
characteristics preserved in the agent's generated code, actual value
distributions in clinical_data/) from SECOND-HAND claims (the previous agent's
`note` / `mapping_note` / `trait_is_control`). The adjudicator is told which is
which and that the second-hand layer is known to be wrong in some cohorts.

Usage:
    python build_packets.py                 # qualified pool only
    python build_packets.py --all           # every cohort in the inventory
"""
import argparse
import glob as _glob
import gzip
import os
import re
import sys
from collections import Counter

import pandas as pd

from grn import paths as config                                  # noqa: E402
# expression matrices live inside the project; raw GEO is optional and only
# needed when re-running adjudication (set GRN_RAW_GEO to point at it)
PRE = config.MATRICES
RAW_GEO = config.RAW_GEO
TF_LIST = config.TF_LIST
OUT = config.out("packets")

MIN_GENES = 15000
MIN_SAMPLES = 30


def slug(s):
    return re.sub(r"[^\w.-]+", "_", s).strip("_")


def _split_row(line):
    """Series-matrix rows are tab-separated with each field quoted."""
    parts = line.rstrip("\n").split("\t")
    return [p.strip().strip('"') for p in parts]


def raw_geo_summary(disease, cohort, max_vals=30):
    """Series title/summary/design plus the full sample-characteristics table,
    read straight from the downloaded series matrix. This is the ground truth
    the previous pipeline was summarising — and sometimes got wrong."""
    d = os.path.join(RAW_GEO, disease, cohort)
    files = sorted(_glob.glob(os.path.join(d, "*series_matrix.txt.gz")))
    if not files:
        return None

    series, chars, n_samples = {}, [], None
    for fp in files:
        try:
            with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("!series_matrix_table_begin"):
                        break
                    if line.startswith("!Series_"):
                        f = _split_row(line)
                        key = f[0][1:]
                        val = " | ".join(x for x in f[1:] if x)
                        series.setdefault(key, [])
                        if val and val not in series[key]:
                            series[key].append(val)
                    elif line.startswith("!Sample_characteristics_ch1"):
                        chars.append(_split_row(line)[1:])
                    elif line.startswith("!Sample_geo_accession"):
                        n_samples = len(_split_row(line)) - 1
        except Exception as e:
            return f"(could not read {os.path.basename(fp)}: {e})"

    out = [f"Series matrix file(s): {', '.join(os.path.basename(f) for f in files)}"]
    if n_samples:
        out.append(f"Samples in the raw series: {n_samples}")
    for k in ("Series_title", "Series_summary", "Series_overall_design",
              "Series_platform_id", "Series_type"):
        if k in series:
            for v in series[k]:
                out.append(f"- **{k}**: {v[:1500]}")
    out.append("")
    out.append("Sample characteristics rows (unique values with sample counts):")
    if not chars:
        out.append("  (none present in the series matrix)")
    for i, row in enumerate(chars):
        c = Counter(x for x in row if x != "")
        shown = "; ".join(f"{k!r}×{v}" for k, v in c.most_common(max_vals))
        if len(c) > max_vals:
            shown += f"; ...(+{len(c)-max_vals} more distinct values)"
        out.append(f"  - row {i}: {shown}")
    return "\n".join(out)


def code_reasoning(disease, cohort, max_lines=170):
    """The agent's generated script, truncated at the point where gene-matrix
    plumbing starts. Everything above it is trait/age/gender reasoning and
    carries the raw sample-characteristics strings in comments."""
    p = f"{PRE}/{disease}/code/{cohort}.py"
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    stop = len(lines)
    for i, ln in enumerate(lines):
        if "Gene Data Extraction" in ln or "Gene Identifier Review" in ln:
            stop = i
            break
    return "".join(lines[:min(stop, max_lines)])


def clinical_summary(disease, cohort, max_vals=25):
    """Actual distribution of every extracted clinical row (post-conversion)."""
    p = f"{PRE}/{disease}/clinical_data/{cohort}.csv"
    if not os.path.isfile(p):
        return None
    try:
        df = pd.read_csv(p, index_col=0)
    except Exception as e:
        return f"(unreadable: {e})"
    out = []
    for row in df.index:
        s = df.loc[row]
        vc = s.value_counts(dropna=False)
        shown = "; ".join(f"{k}={v}" for k, v in list(vc.items())[:max_vals])
        if len(vc) > max_vals:
            shown += f"; ...(+{len(vc)-max_vals} more distinct values)"
        out.append(f"  - `{row}`  n_non_null={s.notna().sum()}/{len(s)}  ->  {shown}")
    return "\n".join(out)


def build(rec, tfs):
    d, c = rec.disease, rec.cohort
    code = code_reasoning(d, c)
    clin = clinical_summary(d, c)

    tf_cov = ""
    try:
        import csv as _csv
        with open(rec.path, "r", encoding="utf-8", newline="") as fh:
            cols = set(next(_csv.reader(fh))[1:])
        n = len(cols & tfs)
        tf_cov = f"{n} / {len(tfs)}  ({n/len(tfs):.1%} of the hg38 TF list)"
    except Exception:
        tf_cov = "(could not read header)"

    P = []
    P.append(f"# Cohort adjudication packet\n")
    P.append(f"**Disease under study:** `{d}`")
    P.append(f"**Cohort:** `{c}`   **Source:** {rec.source}")
    P.append(f"**Expression matrix:** `{rec.path}`")
    P.append(f"**Samples:** {rec.n_samples}   **Genes:** {rec.n_genes}   "
             f"**TF coverage:** {tf_cov}\n")

    P.append("---\n")
    P.append("## A. FIRST-HAND EVIDENCE (trust this)\n")
    raw = raw_geo_summary(d, c) if rec.source != "TCGA" else None
    P.append("### A0. Raw GEO series metadata — the strongest evidence in this packet")
    if raw:
        P.append("Read straight from the downloaded series matrix. It has not passed "
                 "through any agent. Where anything else in this packet disagrees with "
                 "A0, A0 is right.\n")
        P.append("```\n" + raw + "\n```\n")
    elif rec.source == "TCGA":
        P.append("This is a TCGA cohort, not GEO. TCGA samples carry a barcode whose "
                 "sample-type field distinguishes primary tumour from matched normal; "
                 "the trait column is 1 for tumour and 0 for normal.\n")
    else:
        P.append("(raw series matrix not found on disk for this cohort — fall back on "
                 "A1/A2, and consider fetching the GEO page)\n")

    P.append("### A1. The analysis script a previous agent generated for this cohort")
    P.append("Its *comments* quote the raw `!Sample_characteristics_ch1` values verbatim "
             "from the GEO series. Those quoted strings are primary evidence. The agent's "
             "prose conclusions in the same file are not.\n")
    P.append("```python\n" + (code or "(no code file found)") + "\n```\n")

    P.append("### A2. Actual value distribution of every extracted clinical variable")
    P.append("(post-conversion, as stored in `clinical_data/`)\n")
    P.append((clin or "(no clinical_data file found)") + "\n")

    P.append("---\n")
    P.append("## B. SECOND-HAND CLAIMS (verify, do not assume)\n")
    P.append("These were written by the previous pipeline and are **known to be wrong for "
             "some cohorts**. Treat as hypotheses to check against section A.\n")
    P.append(f"- `is_trait_available` = {rec.is_trait_available}")
    P.append(f"- `trait_is_control` = {rec.trait_is_control}")
    P.append(f"- `trait_col` header = `{rec.trait_col}`")
    P.append(f"- `mapping_note` = {rec.mapping_note!r}")
    P.append(f"- `count_0` = {rec.count_0}   `count_1` = {rec.count_1}")
    P.append(f"- `has_age` = {rec.has_age}   `has_gender` = {rec.has_gender}")
    P.append(f"\n- agent's free-text note:\n\n> {str(rec.note)[:2500]}\n")
    return "\n".join(P)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="build packets for every cohort, not just the qualified pool")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    I = pd.read_csv(config.COHORT_INVENTORY)
    if not args.all:
        I = I[(I.n_genes >= MIN_GENES) & (I.n_samples >= MIN_SAMPLES)]
    if args.limit:
        I = I.head(args.limit)

    tfs = {l.strip() for l in open(TF_LIST) if l.strip()}
    os.makedirs(OUT, exist_ok=True)

    n = 0
    for _, r in I.iterrows():
        name = f"{slug(r.disease)}__{slug(r.cohort)}.md"
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            fh.write(build(r, tfs))
        n += 1
    print(f"wrote {n} packets -> {OUT}")


def run(**kw):
    """Entry point for the pipeline runner. Keyword arguments override the
    command-line defaults; the body is the same either way."""
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

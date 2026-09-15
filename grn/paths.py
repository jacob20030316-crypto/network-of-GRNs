"""Every path the project uses, in one place.

Everything resolves under one directory, and every entry can be pointed
somewhere else with an environment variable. Put the data wherever it fits and
say so once:

    export GRN_DATA=/path/to/Data          # everything, in the layout below
    export GRN_RAW_GEO=/somewhere/else     # or move one piece on its own

    Data/
      00_raw/          GEO/ and TCGA/ as downloaded          GRN_RAW
      01_input/        expression matrices from Stage 1      GRN_INPUT
      02_curation/     verdicts, analysis units, tissue      GRN_CURATION
      03_resources/    CollecTRI, TF list, motifs, UBERON    GRN_RESOURCES
      04_external/     Open Targets, ChEA3                   GRN_EXTERNAL
      05_networks/     what the two routes build             written here
      06_readout/      calibration and selection             written here
      07_agent/        evidence packs and interpretations    written here

Reading and writing are kept apart where a stage could otherwise overwrite its
own input: curation reads from GRN_CURATION and writes to 02_curation, so a
re-derived table lands beside the one it should be compared with.

The names exported here are a superset of the older `config.py`, so a script
ported from it needs no change to its body:

    from grn import paths as config

The names exported here are a superset of the older `config.py`, so a script
ported from it needs no change to its body:

    from grn import paths as config
"""
import os

import pandas as pd


def _env(key, default):
    return os.environ.get(key, default)


# --------------------------------------------------------------------------
# where this file sits
# --------------------------------------------------------------------------

PROGRAM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PROGRAM)                       # GRN_Network/

DATA = _env("GRN_DATA", os.path.join(ROOT, "Data"))

# An earlier checkout to compare a run against, for `run.py compare`. Unset by
# default: it exists to check a migration, not to run the pipeline.
LEGACY = _env("GRN_LEGACY", "")

# --------------------------------------------------------------------------
# read-only inputs
# --------------------------------------------------------------------------

# Raw repository downloads. The preprocessing stage expects a directory holding
# GEO/ and TCGA/, and the condition-axis extraction reads the same series
# matrices. `preprocess/download/` fetches them.
RAW = _env("GRN_RAW", os.path.join(DATA, "00_raw"))
RAW_GEO = _env("GRN_RAW_GEO", os.path.join(RAW, "GEO"))
RAW_TCGA = _env("GRN_RAW_TCGA", os.path.join(RAW, "TCGA"))

# Expression matrices from the preprocessing stage, one file per cohort as
# <disease>/<cohort>.csv with samples in rows.
MATRICES = _env("GRN_INPUT", os.path.join(DATA, "01_input"))

# Curation products: cohort verdicts, analysis units, condition axes, tissue.
TABLES = _env("GRN_CURATION", os.path.join(DATA, "02_curation"))

# Fixed resources: CollecTRI, the TF list, motif databases, UBERON.
RESOURCES = _env("GRN_RESOURCES", os.path.join(DATA, "03_resources"))

# External evidence databases: Open Targets, ChEA3.
EXTERNAL = _env("GRN_EXTERNAL", os.path.join(DATA, "04_external"))

# Per-cohort clinical tables and generated scripts kept for provenance.
PROVENANCE = _env("GRN_PROVENANCE", os.path.join(DATA, "02_curation", "provenance"))

# --------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------

# One working directory per construction route. These map one-to-one onto the
# older layout — CollecTRI/Output and De_Novo/Output — so that a ported script
# writes the same file names and `run.py compare` is a directory diff.
# Curation writes here, and reads from TABLES. The two are separate on purpose:
# TABLES defaults to an existing set of tables, and a curation re-run that wrote
# back to where it read would overwrite the very assets it is being checked
# against. Point GRN_CURATION at this directory once a full re-run has produced
# a complete set, and the pipeline then runs on its own curation.
CURATION_OUT = os.path.join(DATA, "02_curation")

NETWORKS = os.path.join(DATA, "05_networks")
SCAFFOLD_OUT = os.path.join(NETWORKS, "scaffold")     # was CollecTRI/Output
DENOVO_OUT = os.path.join(NETWORKS, "denovo")         # was De_Novo/Output
READOUT = os.path.join(DATA, "06_readout")            # was Verification/Output
AGENT_OUT = os.path.join(DATA, "07_agent")            # was Agent/

# Stage checkpoints.
DONE = os.path.join(DATA, ".done")

for _d in (DATA, CURATION_OUT, NETWORKS, SCAFFOLD_OUT, DENOVO_OUT,
           READOUT, AGENT_OUT, DONE):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# individual files
# --------------------------------------------------------------------------

COLLECTRI = os.path.join(RESOURCES, "omnipath_collectri_tf_target.tsv")
TF_LIST = os.path.join(RESOURCES, "allTFs_hg38.txt")
MOTIF_ANNOTATIONS = os.path.join(RESOURCES,
                                 "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")
MOTIF_RANKINGS_GLOB = os.path.join(RESOURCES, "motif_rankings", "hg38*.feather")
UBERON = os.path.join(RESOURCES, "uberon-basic.obo")

COHORT_INVENTORY = os.path.join(TABLES, "cohort_inventory.csv")
COHORT_VERDICTS = os.path.join(TABLES, "cohort_verdicts.csv")
COHORT_TISSUE = os.path.join(TABLES, "cohort_tissue.csv")
NETWORK_YIELD = os.path.join(TABLES, "network_yield.csv")
CONDITION_LEVELS = os.path.join(TABLES, "condition_levels_all.csv")
CONDITION_TCGA = os.path.join(TABLES, "condition_tcga.csv")
ANALYSIS_UNITS = os.path.join(TABLES, "analysis_units.csv")
TISSUE_ONTOLOGY = os.path.join(TABLES, "tissue_ontology.csv")


def out(name):
    """Where a curation product is written, as opposed to read from.

    Reading uses the module-level constants, which point at whatever set of
    tables the run was configured with. Writing always goes here, so a
    re-derived table lands beside the old one instead of on top of it.
    """
    return os.path.join(CURATION_OUT, name)


def matrix_path(disease, cohort):
    """Absolute path of one cohort's expression matrix.

    Resolved here rather than read from the `path` column of the verdict table,
    which records where preprocessing happened to write and is therefore
    specific to one machine.
    """
    return os.path.join(MATRICES, disease, f"{cohort}.csv")


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def require_raw_geo():
    """Abort unless raw GEO is reachable.

    Scripts that scan the series matrices produce an *empty* result when
    RAW_GEO is not set, and would then overwrite a real data asset with that
    emptiness. Call this first so the failure is loud instead of silent.
    """
    import glob
    if not os.path.isdir(RAW_GEO) or not glob.glob(os.path.join(RAW_GEO, "*", "*")):
        raise SystemExit(
            f"raw GEO not found at {RAW_GEO}\n"
            f"  export GRN_RAW_GEO=/path/to/GEO\n"
            f"  (this script rewrites a curation table — refusing to run blind)")


def require_raw_tcga():
    """Same guard for the raw TCGA clinical files."""
    import glob
    if not os.path.isdir(RAW_TCGA) or not glob.glob(os.path.join(RAW_TCGA, "*")):
        raise SystemExit(
            f"raw TCGA not found at {RAW_TCGA}\n"
            f"  export GRN_RAW_TCGA=/path/to/TCGA")


def write_table(df, path, min_rows=1, shrink_tolerance=0.5):
    """Write a data asset, refusing to replace a bigger one with a much smaller one.

    Every curation table is expensive to regenerate, and the usual way to
    destroy one is a re-run that silently produced nothing. Refuse both the
    empty case and the "lost more than half the rows" case unless
    GRN_FORCE_WRITE=1.
    """
    n = len(df)
    if n < min_rows:
        raise SystemExit(f"refusing to write {path}: only {n} rows "
                         f"(expected >= {min_rows}) — the inputs are probably missing")
    if os.path.isfile(path) and os.environ.get("GRN_FORCE_WRITE") != "1":
        old = len(pd.read_csv(path))
        if n < old * shrink_tolerance:
            raise SystemExit(
                f"refusing to write {path}: {n} rows would replace {old} rows.\n"
                f"  if this shrink is intended: GRN_FORCE_WRITE=1 python <script>")
    df.to_csv(path, index=False)
    print(f"wrote {path}  ({n} rows)")


# --------------------------------------------------------------------------
# the curation tables, as objects
# --------------------------------------------------------------------------

def verdicts(usable_only=False):
    """Adjudicated cohort table, with `path` resolved against this checkout."""
    V = pd.read_csv(COHORT_VERDICTS)
    V["path"] = [matrix_path(d, c) for d, c in zip(V.disease, V.cohort)]
    V["usable"] = (V.verdict.isin(["case_control", "all_patients"])
                   & (V.confidence != "low"))
    return V[V.usable].reset_index(drop=True) if usable_only else V


def units(network=None):
    """The analysis-unit registry — the nodes of the two networks."""
    U = pd.read_csv(ANALYSIS_UNITS)
    return U[U.network == network].reset_index(drop=True) if network else U


def sample_overlap(nids):
    """Boolean n x n over analysis units: do these two units share any sample?

    This is the correct test for "these two units are not independent", and
    `cohort_a != cohort_b` is NOT a substitute for it: every TCGA disease carries
    the literal cohort name "TCGA", so a cohort-name test silently discards every
    TCGA-TCGA pair — 42% of the conditioned network's edges — while also missing
    GEO cohorts that were re-used under two disease labels.

    Built through a sample -> units index, so it is linear in samples rather
    than quadratic in units.
    """
    import numpy as np
    from collections import defaultdict
    U = pd.read_csv(ANALYSIS_UNITS).set_index("unit_id")
    pos = {n: i for i, n in enumerate(nids)}
    owners = defaultdict(list)
    for n in nids:
        if n in U.index:
            for s in str(U.loc[n, "samples"]).split("|"):
                owners[s].append(pos[n])
    ov = np.zeros((len(nids), len(nids)), dtype=bool)
    for us in owners.values():
        if len(us) > 1:
            a = np.array(us)
            ov[np.ix_(a, a)] = True
    np.fill_diagonal(ov, False)
    return ov


def tissue():
    return pd.read_csv(COHORT_TISSUE)


def check():
    """Verify the inputs are complete. Returns a list of problems."""
    import glob
    problems = []
    for label, p in [("CollecTRI", COLLECTRI), ("TF list", TF_LIST),
                     ("motif annotations", MOTIF_ANNOTATIONS),
                     ("UBERON", UBERON)]:
        if not os.path.isfile(p):
            problems.append(f"missing {label}: {p}")
    if not glob.glob(MOTIF_RANKINGS_GLOB):
        problems.append(f"missing motif ranking databases: {MOTIF_RANKINGS_GLOB}")
    for label, p in [("cohort inventory", COHORT_INVENTORY),
                     ("cohort verdicts", COHORT_VERDICTS),
                     ("cohort tissue", COHORT_TISSUE),
                     ("analysis units", ANALYSIS_UNITS)]:
        if not os.path.isfile(p):
            problems.append(f"missing {label}: {p}")
    if os.path.isfile(COHORT_VERDICTS):
        V = verdicts()
        miss = [p for p in V.path if not os.path.isfile(p)]
        if miss:
            problems.append(f"{len(miss)}/{len(V)} expression matrices missing, "
                            f"e.g. {miss[0]}")
    return problems


def describe():
    """One screen showing where everything resolved to, and whether it is there."""
    rows = [("raw GEO", RAW_GEO), ("raw TCGA", RAW_TCGA),
            ("expression matrices", MATRICES), ("curation tables", TABLES),
            ("resources", RESOURCES), ("external databases", EXTERNAL),
            ("--- products ---", ""),
            ("scaffold", SCAFFOLD_OUT), ("de novo", DENOVO_OUT),
            ("readout", READOUT), ("agent", AGENT_OUT)]
    for label, p in rows:
        if not p:
            print(f"  {label}")
            continue
        mark = "ok " if os.path.isdir(p) else "MISSING"
        print(f"  {mark:8s} {label:22s} {p}")


if __name__ == "__main__":
    import glob
    print(f"PROGRAM = {PROGRAM}")
    print(f"DATA    = {DATA}\n")
    describe()
    print()
    probs = check()
    if probs:
        print("problems:")
        for p in probs:
            print(f"  x {p}")
        raise SystemExit(1)
    V, U = verdicts(), units()
    print("inputs complete")
    print(f"  adjudicated cohorts  {len(V)}   usable {int(V.usable.sum())} / "
          f"{V[V.usable].disease.nunique()} diseases")
    print(f"  analysis units       {len(U)}   uncond {int((U.network=='uncond').sum())} / "
          f"cond {int((U.network=='cond').sum())}")
    print(f"  motif rankings       {len(glob.glob(MOTIF_RANKINGS_GLOB))}")

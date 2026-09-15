"""The pipeline, declared once.

Reading this file top to bottom is meant to be the same thing as reading the
Methods: what happens, in what order, what each step leaves behind, and which
steps are a judgement rather than a calculation.

Four phases. Each is a list of stages, and a stage is a call plus the files that
call must produce. `grn.runner` executes them, skips what is already done, and
invalidates everything downstream of a step that re-ran.

    curate      what the cohorts are, and how their patients divide
    scaffold    score a curated catalogue per patient group, join the groups
    denovo      infer regulons per disease without a catalogue, join them
    interpret   retrieve external evidence, write the mechanism

`agent=True` marks the steps where a model decides something. They are the
steps whose input is different for every item — whether this cohort holds real
cases, what this study's variables mean, what the literature says about this
factor. Everything else applies one rule to all of them, and is code.

Two stages are not here, because they do not belong in a run that can be
restarted casually. Cohort preprocessing is its own program with its own
checkpoints (`preprocess/`), and de novo inference takes two days
(`scripts/infer_denovo.py`); the de novo phase begins from what it leaves.
"""
import os

from grn import paths
from grn.agents import axes, biologist, curator, network, tissue
from grn.agents import assistant as evidence_agent
from grn.runner import Stage
from grn.tools.curation import units
from grn.tools.denovo import pair_factors, pair_network
from grn.tools.scaffold import (diagnostics, edges, mechanism,
                                method_comparison, null, reliability, vectors)

# --------------------------------------------------------------------------
# curate — from downloaded cohorts to the patient groups everything is built on
# --------------------------------------------------------------------------

CURATE = [
    Stage(
        "admit", curator.run, per_net=False, agent=True,
        produces=["curation:cohort_verdicts.csv"],
        note="does this cohort hold real cases of the disease it is filed under",
    ),
    Stage(
        "axes", axes.run, per_net=False, agent=True,
        produces=["curation:condition_axes.csv",
                  "curation:condition_axes_rejected.csv"],
        note="which recorded variables can divide these patients, and how",
    ),
    Stage(
        "tissue", tissue.run, per_net=False, agent=True,
        produces=["curation:cohort_tissue.csv", "curation:tissue_ontology.csv"],
        note="what material was measured, anchored to an ontology class",
    ),
    Stage(
        "units", units.run, per_net=False,
        produces=["curation:analysis_units.csv",
                  "curation:analysis_units_dropped.csv"],
        note="apply the axis rules to the samples; one group per axis level",
    ),
]

# --------------------------------------------------------------------------
# scaffold — score a fixed catalogue in every patient group
# --------------------------------------------------------------------------

SCAFFOLD = [
    Stage(
        "vectors", vectors.run, per_net=False,
        produces=["scaffold:edge_universe.csv", "scaffold:W_full.npy",
                  "scaffold:A_avail.npy", "scaffold:W_meta.csv"],
        note="score every catalogued interaction in every patient group",
    ),
    Stage(
        "reliability", reliability.run, per_net=False,
        produces=["scaffold:reliability_by_m.csv",
                  "scaffold:reliability_halfsplit.csv"],
        note="split-half reliability, used to disattenuate the similarity",
    ),
    Stage(
        "diagnostics", diagnostics.run, per_net=False,
        produces=["scaffold:meta_uncond.csv", "scaffold:meta_cond.csv",
                  "readout:acceptance_uncond.csv", "readout:acceptance_cond.csv"],
        note="the correction ladder and the four confound checks",
    ),
    Stage(
        "similarity", method_comparison.run, net_kw="network",
        params={"core": True, "ladder": "full"},
        produces=["scaffold:S_M3_S3signed_{net}_core.npy",
                  "readout:similarity_gate_{net}_core.csv"],
        note="six similarity definitions on one ruler; the winner is kept",
    ),
    Stage(
        "inspect", network.run, agent=True,
        params={"route": "scaffold", "inspect_only": True},
        produces=["readout:inspection/scaffold_{net}.json"],
        note="may this network be used, given the checks and the discrimination",
    ),
    Stage(
        "edges", edges.run, params={"perms": 300},
        produces=["scaffold:edges_{net}.csv", "scaffold:disease_pairs_{net}.csv"],
        note="apply the joining rules, calibrate against permuted labels",
    ),
    Stage(
        "null", null.run, params={"shuffles": 400},
        produces=["readout:mechanism_null_{net}.csv",
                  "readout:mechanism_null_draws_{net}.npz"],
        note="matched permutation null for the pair and for each rank",
    ),
    Stage(
        "mechanism", mechanism.run,
        params={"emit_agent": True,
                "pairs_from": os.path.join(paths.READOUT,
                                           "mechanism_null_{net}.csv"),
                "q_col": "q_joint", "q_max": 0.05},
        produces=["scaffold:mechanism_pairs_{net}.csv",
                  "scaffold:agent_prompts_{net}.jsonl"],
        note="read out the factors of every pair that survived the null",
    ),
]

# --------------------------------------------------------------------------
# denovo — infer regulons per disease, with no catalogue consulted
# --------------------------------------------------------------------------

DENOVO = [
    Stage(
        "similarity", pair_network.run, per_net=False, params={"compare": True},
        produces=["denovo:denovo_method_comparison.csv"],
        note="five ways to compare two inferred networks, on one ruler",
    ),
    Stage(
        "inspect", network.run, per_net=False, agent=True,
        params={"route": "denovo", "net": "uncond", "inspect_only": True},
        produces=["readout:inspection/denovo_uncond.json"],
        note="may this network be used, given what it is used for",
    ),
    Stage(
        "pairs", pair_network.run, per_net=False,
        params={"perms": 500, "top_pairs": 40, "top_tfs": 12, "emit_agent": True},
        produces=["denovo:denovo_pairs.csv", "denovo:denovo_pair_mechanisms.csv",
                  "denovo:denovo_pair_summary.csv",
                  "denovo:denovo_agent_prompts.jsonl"],
        note="disease-pair layer and the regulons each pair shares",
    ),
    Stage(
        "factors", pair_factors.run, per_net=False,
        produces=["denovo:denovo_pair_factors_all.csv"],
        note="every pair, every shared regulon, against that factor's own background",
    ),
]

# --------------------------------------------------------------------------
# interpret — what is on record, and what it means
# --------------------------------------------------------------------------

INTERPRET = [
    Stage(
        "evidence", evidence_agent.run, per_net=False, agent=True,
        produces=["agent:evidence"],
        note="retrieve Open Targets and Europe PMC for every reported factor",
    ),
    Stage(
        "interpret", biologist.run, per_net=False, agent=True,
        produces=["agent:interpretations"],
        note="write the mechanism and judge how far the evidence carries it",
    ),
]

PHASES = {"curate": CURATE, "scaffold": SCAFFOLD,
          "denovo": DENOVO, "interpret": INTERPRET}

# The order `--all` runs them in.
ORDER = ["curate", "scaffold", "denovo", "interpret"]


def precheck(phase):
    """What must already exist before a phase can start."""
    import glob
    if phase == "scaffold" or phase == "curate":
        if not os.path.isdir(paths.MATRICES) or not glob.glob(
                os.path.join(paths.MATRICES, "*", "*.csv")):
            return (f"no preprocessed expression matrices under {paths.MATRICES}\n"
                    f"  they come from the preprocessing stage, run separately:\n"
                    f"      cd preprocess && python main.py --data-root "
                    f"{paths.RAW} --version v1 --model gpt-4.1 --skip-filter")
    if phase == "denovo":
        n = len([p for p in glob.glob(os.path.join(paths.DENOVO_OUT,
                                                   "*.regulons.tsv"))
                 if not os.path.basename(p).split(".")[0].endswith(
                     tuple(f"__r{i}" for i in range(10)))])
        if n == 0:
            return (f"no inferred regulons in {paths.DENOVO_OUT}\n"
                    f"  this phase starts from the two-day inference, run "
                    f"separately:\n"
                    f"      nohup python scripts/infer_denovo.py --repeats 3 &")
    return None

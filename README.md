# A network of gene regulatory networks

Code for *Large-scale agentic analysis identifies recurrent regulators linking
diseases across patient subgroups*.

Diseases that appear unrelated clinically can share molecular programmes.
Comparing diseases through gene regulatory networks (GRNs) identifies the
regulators through which those programmes converge, but a regulatory network is
not a fixed property of a disease: it changes with the patients from which it is
estimated. This framework uses that variation as a criterion. It estimates a
regulatory network for every clinically defined patient subgroup in public
cohorts, joins those networks into a network of regulatory networks, and
prioritises the regulators that recur as the sampled patients and recorded
conditions change.

Starting from transcriptomic cohorts spanning 132 diseases in GEO and TCGA, the
framework resolved recorded clinical characteristics into 86 comparable axes
and compared 212,469 pairs of subgroup-level regulatory networks across 2,145
cross-disease pairs. After false-discovery-rate correction, 222 disease pairs
combined recurrent leading regulators with reproducible regulator rankings.

## Overview

The pipeline has three stages and four agents.

| Stage | What it does | Agent |
|---|---|---|
| 1. Preprocessing and curation | Converts each GEO/TCGA cohort into an expression matrix aligned with its clinical annotation, admits cohorts that contain genuine cases, resolves recorded clinical variables into axes, and divides patients into groups | Preprocessing agent |
| 2. Network construction | Builds the networks of regulatory networks through a curated-scaffold route and a de novo route, and selects disease pairs and regulators against permuted disease labels | Network-of-networks agent |
| 3. Interpretation | Retrieves evidence from Open Targets and Europe PMC, then writes a mechanistic account of each retained disease pair and judges its support | Assistant agent, biologist agent |

The agents make the judgements that change from item to item: whether a cohort
holds real cases, what a study's clinical variables mean, whether a constructed
network may be read out, and what the literature says about a factor. Network
construction, similarity, calibration and the permutation null apply one fixed
procedure to every patient group and are code.

### The two construction routes

| | Patients partitioned by clinical axis | Patients pooled within a cohort |
|---|---|---|
| Curated CollecTRI scaffold | 663 nodes | 111 nodes |
| De novo inference | — | 48 nodes |

**Curated scaffold.** The 44,350 transcription factor–target interactions in
CollecTRI are weighted within each patient group (≥30 samples; pooled cohorts
≥50 samples) by the Pearson correlation between factor and target. Interactions
estimable in at least 99% of groups are retained (37,500 interactions, 837
factors). Node similarity is the mean, over factors, of the agreement between
their normalised target-weight vectors, corrected for split-half reliability and
centred within source class. For each disease pair, every factor is scored in
every combination of one patient group from each disease; its *recurrence* is
the fraction of combinations in which it reaches the 90th percentile of its
tissue-matched background. Disease pairs are tested with a joint permutation
statistic on recurrence strength and split-half ranking agreement, adjusted by
Benjamini–Hochberg; 222 of 485 tested pairs pass at q_joint < 0.05.

**De novo.** Within each cohort of ≥100 samples, 100 samples are drawn three
times and regulons are inferred with GRNBoost2 and pySCENIC (motif NES ≥ 3,
present in at least two of three draws). No curated interactions are used.
Nodes are compared by the enrichment-weighted Jaccard overlap of regulons, and a
disease pair is retained when at least three factors share more targets than
that factor's own cross-disease background (within-pair FDR 0.05).

## Installation

```bash
git clone https://github.com/jacob20030316-crypto/network-of-GRNs.git
cd network-of-GRNs
conda env create -f environment.yml
conda activate grn
```

De novo inference additionally needs `pyscenic` and `arboreto`, which pin older
versions of numpy and pandas; install them in a separate environment if they
clash (see `environment.yml`). Preprocessing has its own requirements in
`preprocess/requirements.txt`.

## Data

### Transcriptomic cohorts

The GEO and TCGA cohorts are available from Google Drive:

**https://drive.google.com/drive/folders/1kxHOyW5wNnY3Rk15xwLaM7ZZS01wGzRO?usp=drive_link**

Download both folders, `GEO/` and `TCGA/`, into one directory and point the
pipeline at it:

```bash
export GRN_RAW=/path/to/raw          # the directory holding GEO/ and TCGA/
export GRN_DATA=/path/to/output      # where everything is written; default ./Data
```

### Reference resources

Place these under `$GRN_DATA/03_resources/`:

| File | Source |
|---|---|
| `omnipath_collectri_tf_target.tsv` | CollecTRI, via OmniPath |
| `allTFs_hg38.txt` | human TF list, cisTarget resources |
| `motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl` | motif annotations, cisTarget resources |
| `motif_rankings/hg38*.feather` | motif ranking databases, cisTarget resources |
| `uberon-basic.obo` | UBERON ontology |

and these under `$GRN_DATA/04_external/`:

| Directory | Contents |
|---|---|
| `opentargets/` | Open Targets Platform `target/`, `disease/` and `association_overall_direct/` parquet files |
| `chea3/` | ChEA3 GMT libraries, including `ReMap_ChIP-seq.gmt` |

Check what is present:

```bash
python run.py paths
```

## Language models

The agents call a model through an API. Two backends are supported:

```bash
# OpenAI, or any OpenAI-compatible endpoint
export GRN_LLM=openai
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...           # optional, for a compatible endpoint

# Anthropic
export GRN_LLM=anthropic
export ANTHROPIC_API_KEY=...
```

`GRN_LLM_MODEL` sets the model for every agent; `--backend` and `--model`
override both for a single agent. The models used in the paper were:

| Agent | Model |
|---|---|
| Preprocessing agent | GPT-4.1 |
| Network-of-networks agent | Claude Opus 5 |
| Assistant agent | GPT-5.6 |
| Biologist agent | Claude Opus 5 |

Preprocessing reads its keys from `preprocess/.env`; copy
`preprocess/env.example` and fill in the providers you use.

## Running the pipeline

```bash
python run.py plan        # every stage, and which ones call an agent
python run.py all         # the whole pipeline
python run.py status      # what has been built
```

Two steps run separately, before the phase that needs them:

```bash
# Stage 1: cohorts -> expression matrices
cd preprocess
python main.py --data-root $GRN_RAW --version v1 --model gpt-4.1 \
    --skip-filter --parallel-mode cohorts --max-workers 8
cd ..

# De novo regulon inference (about two days)
nohup python scripts/infer_denovo.py --repeats 3 &
```

Phases can also be run one at a time:

```bash
python run.py curate                    # admission, clinical axes, tissue, patient groups
python run.py scaffold --net cond       # curated scaffold, patient-group network
python run.py scaffold --net uncond     # curated scaffold, pooled-cohort network
python run.py denovo                    # de novo route
python run.py interpret                 # evidence retrieval and mechanistic accounts
```

and stages within a phase with `--only <stage>` or `--from <stage>`. A stage
whose outputs exist and whose parameters are unchanged is skipped; a stage that
re-runs invalidates every stage after it. `--force` re-runs regardless and
`--dry-run` shows what would run.

Each agent can also be run on its own, for example to choose a backend and model
per agent:

```bash
python -m grn.agents.curator   --backend openai    --model gpt-4.1 --jobs 8
python -m grn.agents.axes      --backend openai    --model gpt-4.1 --jobs 8
python -m grn.agents.tissue    --backend openai    --model gpt-4.1 --jobs 8
python -m grn.agents.network   --backend anthropic --route scaffold --net cond
python -m grn.agents.assistant --backend openai    --jobs 6
python -m grn.agents.biologist --backend anthropic --jobs 6
```

Each agent writes one file per item and skips items already written, so an
interrupted run resumes. Items that fail are recorded in `failures.jsonl` with
the reason and do not stop the batch. Two checks are applied to every reply: an
agent may name only the factors it was given, and may cite only identifiers that
appear in its evidence.

## Outputs

```
Data/
  02_curation/     cohort admission, clinical axes, tissue, patient groups
  05_networks/     the curated-scaffold and de novo networks
  06_readout/      calibration, permutation null, network inspection
  07_agent/        evidence packs and mechanistic accounts
```

## Results included

`Data/interpretations/` contains the mechanistic accounts of the 222 disease
pairs, one JSON file per pair: the shared programme, the factors carrying it,
its role in each disease, a verdict, the strength of the retrieved support, the
PMIDs relied on, and an experiment that would falsify it. Field definitions are
in `Data/interpretations/README.md`. The accounts are hypotheses; none has been
tested experimentally.

## Repository layout

```
run.py                     entry point
grn/pipeline.py            the four phases and their stages
grn/runner.py              stages, checkpoints, resumption
grn/paths.py               every input and output path, overridable by environment variable
grn/agents/                agents, their prompts, and the model interface (llm.py)
grn/tools/curation/        cohort reading and patient-group construction
grn/tools/scaffold/        curated-scaffold route
grn/tools/denovo/          de novo route
grn/tools/evidence.py      Open Targets, Europe PMC and ChIP evidence
preprocess/                Stage 1 preprocessing, adapted from GenoMAS
scripts/infer_denovo.py    de novo regulon inference
```

## Licence

MIT; see `LICENSE`. `preprocess/` is adapted from
[GenoMAS](https://github.com/Liu-Hy/GenoMAS) (MIT); its attribution and the list
of changes are in `preprocess/README.md`.

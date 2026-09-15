# Large-scale agentic analysis identifies recurrent regulators linking diseases across patient subgroups

This repository contains the code for the paper "Large-scale agentic analysis
identifies recurrent regulators linking diseases across patient subgroups."

The pipeline uses language-model agents to curate public transcriptomic cohorts,
builds a gene regulatory network for every clinically defined patient subgroup,
joins those networks into a network of regulatory networks, and interprets the
regulators that recur across patient subgroups of two diseases.

## Overview

A regulatory network estimated once from a whole cohort can hide signals that
appear only in particular patient subgroups. This workflow uses the variation
across subgroups as a criterion, by:

- Preprocessing GEO and TCGA cohorts and resolving their recorded clinical
  variables into patient groups
- Building networks of regulatory networks through a curated-scaffold route
  (CollecTRI) and a de novo route (GRNBoost2 and pySCENIC)
- Selecting disease pairs and regulators by how often regulators recur across
  patient-group comparisons, against permuted disease labels
- Retrieving evidence from Open Targets and Europe PMC and writing a mechanistic
  interpretation for each retained disease pair

The analysis is driven by four agents: a preprocessing agent, a
network-of-networks agent, an assistant agent and a biologist agent.

## Repository structure

```
.
├── run.py                    # Main entry point
├── environment.yml           # Conda environment
├── grn/
│   ├── pipeline.py           # The four phases and their stages
│   ├── runner.py             # Stage execution, checkpoints and resumption
│   ├── paths.py              # Input and output paths
│   ├── agents/               # Agents, their prompts, and the model interface
│   └── tools/
│       ├── curation/         # Cohort reading and patient-group construction
│       ├── scaffold/         # Curated-scaffold route
│       ├── denovo/           # De novo route
│       └── evidence.py       # Open Targets, Europe PMC and ChIP evidence
├── preprocess/               # Cohort preprocessing, adapted from GenoMAS
├── scripts/
│   └── infer_denovo.py       # De novo regulon inference
└── Data/
    └── interpretations/      # Mechanistic interpretations of the reported disease pairs
```

## Requirements

- Python 3.10
- An OpenAI or Anthropic API key
- pySCENIC and arboreto for de novo inference, preferably in a separate
  environment (see `environment.yml`)

```bash
conda env create -f environment.yml
conda activate grn
```

Preprocessing has its own dependencies in `preprocess/requirements.txt`.

## Dataset

The GEO and TCGA cohorts are available on Google Drive:

[Download dataset](https://drive.google.com/drive/folders/1kxHOyW5wNnY3Rk15xwLaM7ZZS01wGzRO?usp=drive_link)

| Item | Description |
|---|---|
| `GEO/` | GEO transcriptomic cohorts |
| `TCGA/` | TCGA transcriptomic cohorts |

Download both folders into one directory and set:

```bash
export GRN_RAW=/path/to/raw        # directory holding GEO/ and TCGA/
export GRN_DATA=/path/to/Data      # where outputs are written (default: ./Data)
```

`python run.py paths` shows where every input and output resolves to.

## Language models

Set one of the following:

```bash
# OpenAI or any OpenAI-compatible endpoint
export GRN_LLM=openai
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=your_endpoint    # optional

# Anthropic
export GRN_LLM=anthropic
export ANTHROPIC_API_KEY=your_key
```

`GRN_LLM_MODEL` selects the model. In the paper, the preprocessing agent used
GPT-4.1, the assistant agent used GPT-5.6, and the network-of-networks and
biologist agents used Claude Opus 5. Preprocessing reads its keys from
`preprocess/.env` (copy `preprocess/env.example`).

## Quick start

```bash
# 1. Preprocess cohorts into expression matrices
cd preprocess
python main.py --data-root $GRN_RAW --version v1 --model gpt-4.1 \
    --skip-filter --parallel-mode cohorts --max-workers 8
cd ..

# 2. Infer de novo regulons (long-running)
nohup python scripts/infer_denovo.py --repeats 3 &

# 3. Run the pipeline
python run.py all
```

`python run.py plan` lists every stage and marks the ones that call an agent;
`python run.py status` shows what has been built.

## Pipeline steps

Each phase can be run on its own, for example `python run.py scaffold --net cond`.
Stages whose outputs already exist are skipped, so an interrupted run resumes.

**Curate** (`python run.py curate`)
- Admits cohorts that contain cases of the disease they are filed under
- Resolves recorded clinical variables into clinical axes and tissue
- Divides patients into groups along those axes
- Writes to `Data/02_curation/`

**Scaffold** (`python run.py scaffold --net cond` or `--net uncond`)
- Weights CollecTRI interactions within each patient group (`cond`) or pooled cohort (`uncond`)
- Builds the network of regulatory networks and checks it for confounding
- Calibrates disease pairs and regulators against permuted disease labels
- Writes to `Data/05_networks/scaffold/` and `Data/06_readout/`

**De novo** (`python run.py denovo`)
- Compares regulons inferred without a curated scaffold
- Tests shared targets of each regulator against that regulator's own background
- Writes to `Data/05_networks/denovo/`

**Interpret** (`python run.py interpret`)
- Retrieves Open Targets and Europe PMC evidence for the reported regulators
- Writes a mechanistic interpretation for each disease pair
- Writes to `Data/07_agent/`

## Outputs

| Directory | Contents |
|---|---|
| `Data/02_curation/` | Cohort admission, clinical axes, tissue and patient groups |
| `Data/05_networks/` | Curated-scaffold and de novo networks |
| `Data/06_readout/` | Calibration, permutation null and network inspection |
| `Data/07_agent/` | Evidence and mechanistic interpretations |

## License

This project is licensed under the MIT License — see `LICENSE`. `preprocess/` is
adapted from [GenoMAS](https://github.com/Liu-Hy/GenoMAS) (MIT); see
`preprocess/README.md`.

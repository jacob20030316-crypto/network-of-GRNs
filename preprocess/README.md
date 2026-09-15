# Stage 1 — preprocessing

The first stage of the pipeline. It takes the raw GEO and TCGA downloads and
returns one expression matrix per cohort, on which everything downstream is
built.

The work is done by agents rather than a fixed script because the input does
not have a fixed shape. Every GEO submission stores its samples, its gene
identifiers and its clinical annotation in its own way, so the code that reads
one cohort is not the code that reads the next. An agent writes that code per
cohort, runs it, reads the error when it fails, and revises — which is what
makes several hundred cohorts processable under one procedure.

## Provenance

This directory is a modified copy of **GenoMAS**
(https://github.com/Liu-Hy/GenoMAS, MIT licence, `LICENSE` kept alongside),
from Liu, Li and Wang, *GenoMAS: A Multi-Agent Framework for Scientific
Discovery via Code-Driven Gene Expression Analysis* (arXiv:2507.21035). The
upstream framework supplies the agent architecture: the PI, GEO and TCGA
agents, the code reviewer and domain expert, the notebook-style action units,
and the messaging and execution machinery.

What this copy adds or changes:

    agents/filter_agent.py            } a filtering stage, with its tools,
    tools/filter.py                   } prompts and action units
    prompts/filter.py                 }
    prompts/action_units/base/filter_action_units.json
    run_filter_55.py, summarize_filter_55.py

    tools/preprocess.py               revised preprocessing helpers, with
    tools/preprocess_{8.2,8.5,9.3}.py earlier revisions kept for reference
    prompts/{GEO,TCGA,shared}.py      revised guidelines
    prompts/action_units/base/{geo,tcga}_action_units.json
    environment.py, main.py           orchestration changes, including
    utils/config.py                   the --skip-filter switch
    core/{au_generator,messaging,prompt_loader}.py
    utils/{llm,path_config,utils}.py
    metadata/GSE16717_probe_accession_for_symbol_mapping.csv

## Running it

The raw downloads are a prerequisite, not a stage: a directory holding `GEO/`
and `TCGA/`, as `download/download_GEO_data.py` and `download/geo_manifest.json`
produce it.

    python main.py \
        --data-root /path/to/raw \
        --version v1 \
        --model gpt-4.1 \
        --skip-filter \
        --parallel-mode cohorts --max-workers 8

`--skip-filter` stops after preprocessing. The filter stage belongs to the
upstream task of gene–trait association and splits each cohort along age and
sex only; this project needs its patient groups defined along the 86 clinical
axes each study actually recorded, which is done afterwards by
`grn.tools.curation`, not here.

Output lands in `output/<version>/preprocess/<disease>/<cohort>.csv`, one row
per sample. That directory is what `GRN_INPUT` should point at, and the rest of
the pipeline reads nothing else from this stage.

Progress is checkpointed per cohort, so an interrupted run resumes.

## Keys

Copy `env.example` to `.env` and fill in the provider keys. `.env` is
deliberately absent from this repository and must not be committed.

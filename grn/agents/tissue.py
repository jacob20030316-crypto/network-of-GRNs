"""Stage 1: the tissue agent.

The admission agent records what each cohort measured in the words of whoever
described it. Across 244 cohorts that gives 222 distinct strings — "bone marrow
cells (blasts >=80%)", "brain, middle temporal gyrus", "scalp skin biopsy",
"unknown (raw field only states 'tissue source: patient with AML')". Tissue is
the confound that has to be regressed on, and 222 free-text strings cannot be.

This resolves each into an organ label, a material type, and a UBERON class.

The ontology term is the reason this is worth doing rather than approximating.
UBERON carries a hierarchy, so an organ label anchored to a class comes with its
organ system for free, and the label is a public identifier that someone without
this code can check. What the agent supplies is the name of the class; the id
is looked up here, and a name that does not resolve to exactly one class is
rejected and asked again. So the ontology cannot be hallucinated: an unresolved
name never reaches the table.

This replaces a hand-built vocabulary of ninety-odd lines.
"""
import json
import os

import pandas as pd

from grn import paths
from grn.agents import llm
from grn.tools.curation import ontology as obo

PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "prompts", "tissue.md")
OUT_DIR = os.path.join(paths.CURATION_OUT, "tissue")

SAMPLE_TYPES = {"bulk_tissue", "whole_blood", "pbmc", "sorted_cells",
                "cell_line", "primary_culture", "unknown"}

_terms = None


def ontology():
    """The UBERON classes, parsed once."""
    global _terms
    if _terms is None:
        if not os.path.isfile(paths.UBERON):
            raise SystemExit(f"no UBERON ontology at {paths.UBERON}")
        _terms = obo.parse_obo(paths.UBERON)
    return _terms


def resolve_name(label):
    """The UBERON id for a class name, or None where it is not exactly one."""
    hits = [i for i, t in ontology().items() if t["name"] == label]
    return hits[0] if len(hits) == 1 else None


def brief(row):
    return "\n".join([
        f"disease: {row.disease}",
        f"cohort: {row.cohort}  ({row.source})",
        f"samples: {row.get('case_n') or row.get('n_samples') or '?'}",
        "",
        "how the material was described:",
        f"  {row.tissue}",
        "",
        f"what the cohort was admitted as: {row.verdict}",
    ])


def validator(row):
    def check(obj):
        llm.check_keys(obj, ["disease", "cohort", "tissue_class",
                             "sample_type", "uberon_name", "mixed", "note"])
        if str(obj["cohort"]) != str(row.cohort):
            raise llm.LLMError(f"cohort should be {row.cohort!r}")
        if obj["sample_type"] not in SAMPLE_TYPES:
            raise llm.LLMError(f"sample_type must be one of {sorted(SAMPLE_TYPES)}")
        tc = str(obj["tissue_class"]).strip()
        if not tc:
            raise llm.LLMError("tissue_class is required; use 'unknown' where "
                               "the description names no material")
        name = str(obj["uberon_name"]).strip()
        if tc == "unknown":
            if name:
                raise llm.LLMError("uberon_name must be empty when the tissue "
                                   "class is unknown")
            obj["uberon_id"] = ""
        else:
            if not name:
                raise llm.LLMError("uberon_name is required unless the tissue "
                                   "class is unknown")
            uid = resolve_name(name)
            if uid is None:
                raise llm.LLMError(
                    f"{name!r} is not the name of exactly one UBERON class. "
                    f"Give the class name as UBERON spells it.")
            obj["uberon_id"] = uid
        obj["disease"] = row.disease
        obj["source"] = row.source
        return obj
    return check


def collect(out_dir=OUT_DIR):
    """Gather the decisions, roll the ontology up, and write the tissue table."""
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        if os.path.basename(p) == "failures.jsonl":
            continue
        rows.append(json.load(open(p, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no tissue decisions in {out_dir}")
    T = pd.DataFrame(rows)

    # The organ system comes from the ontology, not from us: each term's
    # is_a/part_of closure is walked up to the systems UBERON defines.
    terms = ontology()
    systems, ancestors_col = [], []
    for uid in T.uberon_id:
        if not uid:
            systems.append("unassigned")
            ancestors_col.append("")
            continue
        anc = obo.ancestors(terms, uid)
        names = [terms[a]["name"] for a in anc if a in terms]
        sysn = [n for n in names if n.endswith("system")
                and n not in obo.SYSTEM_WEAK]
        systems.append(sysn[0] if sysn else "unassigned")
        ancestors_col.append("|".join(names))
    T["organ_system"] = systems
    T["uberon_ancestors"] = ancestors_col
    T["uberon_name"] = [terms[u]["name"] if u else "" for u in T.uberon_id]

    keep = ["disease", "cohort", "source", "tissue_class", "sample_type",
            "uberon_id", "uberon_name", "organ_system", "mixed", "note"]
    paths.write_table(T[keep], paths.out("cohort_tissue.csv"), min_rows=50)
    paths.write_table(
        T[["uberon_id", "uberon_name", "organ_system", "uberon_ancestors"]]
        .drop_duplicates("uberon_id"),
        paths.out("tissue_ontology.csv"), min_rows=1)

    print(f"\n  {T.tissue_class.nunique()} tissue classes over {len(T)} cohorts")
    print(f"  {int((T.tissue_class == 'unknown').sum())} cohorts with no material named")
    print("\n=== sample type ===")
    print(T.sample_type.value_counts().to_string())
    print("\n=== organ system ===")
    print(T.organ_system.value_counts().head(12).to_string())
    return T


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--only", nargs="+", metavar="DISEASE__COHORT")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args(argv)

    if args.collect_only:
        collect(args.out_dir)
        return 0

    ontology()                                   # fail early if it is missing
    V = paths.verdicts(usable_only=True)
    V = V[V.tissue.notna()]
    items = [r for _, r in V.iterrows()]
    if args.only:
        want = set(args.only)
        items = [r for r in items if f"{r.disease}__{r.cohort}" in want]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} admitted cohorts, "
          f"{len(set(r.tissue for r in items))} distinct descriptions")

    prompt = open(PROMPT, encoding="utf-8").read()
    llm.run_batch(items, key_of=lambda r: f"{r.disease}__{r.cohort}",
                  prompt_of=lambda r: prompt + "\n\n---\n\n" + brief(r),
                  validate_of=validator,
                  out_dir=args.out_dir, label="cohort",
                  model=args.model, backend=args.backend, jobs=args.jobs,
                  redo=args.redo)
    print()
    collect(args.out_dir)
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
    import sys
    sys.exit(main())

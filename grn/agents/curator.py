"""Stage 1's second task: the admission agent.

GenoMAS turns each downloaded cohort into an expression matrix. It does not
decide whether that cohort has anything to do with the disease it is filed
under, and a surprising number do not: of 443 cohorts adjudicated here, 95
studied a different condition and 55 held no patients at all — cell lines,
healthy donors, in-vitro drug arms. Left in, each of those contributes a
regulatory network that is then compared against real ones.

The judgement cannot be made from the expression matrix, because by then the
evidence is gone. It has to be made from the original submission: the series
title, the design paragraph, the sample characteristics rows. So the packet the
agent reads is assembled from the raw GEO record, and it separates first-hand
evidence from what the preprocessing pipeline concluded — that second layer is
known to be wrong in some cohorts, which is the reason this step exists.

This replaces a shell script that fanned the same prompt out to `claude -p`.
The prompt and the packets are unchanged; what moves is the surrounding
machinery — retries, schema checking, failure recording, resumption — which is
now the same for every agent in the pipeline.
"""
import json
import os

import pandas as pd

from grn import paths
from grn.agents import llm
from grn.tools.curation import packets

PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "prompts", "admission.md")
OUT_DIR = os.path.join(paths.CURATION_OUT, "verdicts")

VERDICTS = {"case_control", "all_patients", "subgroup_only",
            "not_patients", "wrong_disease", "unclear"}
CONFIDENCE = {"high", "medium", "low"}
MATCH = {"exact", "related", "mismatch"}
PLATFORM = {"genome_wide", "targeted_panel", "unknown"}


def validator(disease, cohort):
    def check(obj):
        llm.check_keys(obj, ["disease", "cohort", "verdict", "case_selector",
                             "n_case_expected", "trait_meaning", "tissue",
                             "platform_type", "disease_match", "confidence",
                             "evidence", "concerns", "second_hand_was_wrong"])
        if obj["verdict"] not in VERDICTS:
            raise llm.LLMError(f"verdict must be one of {sorted(VERDICTS)}")
        if obj["confidence"] not in CONFIDENCE:
            raise llm.LLMError(f"confidence must be one of {sorted(CONFIDENCE)}")
        if obj["disease_match"] not in MATCH:
            raise llm.LLMError(f"disease_match must be one of {sorted(MATCH)}")
        if obj["platform_type"] not in PLATFORM:
            raise llm.LLMError(f"platform_type must be one of {sorted(PLATFORM)}")
        # The packet names the cohort; an answer about a different one means the
        # reply drifted, and silently keeping it would misfile a verdict.
        if str(obj["cohort"]) != str(cohort):
            raise llm.LLMError(f"cohort should be {cohort!r}, got {obj['cohort']!r}")
        if not str(obj.get("evidence", "")).strip():
            raise llm.LLMError("evidence is required: quote the decisive raw "
                               "strings from the packet")
        obj["disease"] = disease
        return obj
    return check


def collect(verdict_dir=OUT_DIR, inventory=None):
    """Gather the per-cohort verdicts into one table, joined onto the inventory."""
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(verdict_dir, "*.json"))):
        if os.path.basename(p) == "failures.jsonl":
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print(f"  ! unreadable: {os.path.basename(p)} ({e})")
            continue
        d["_packet"] = os.path.basename(p).replace(".json", "")
        rows.append(d)
    if not rows:
        raise SystemExit(f"no verdicts in {verdict_dir}")
    V = pd.DataFrame(rows)
    I = pd.read_csv(inventory or paths.COHORT_INVENTORY)
    M = I.merge(V, on=["disease", "cohort"], how="right", suffixes=("", "_v"))
    out = paths.out("cohort_verdicts.csv")
    paths.write_table(M, out, min_rows=100)
    usable = M.verdict.isin(["case_control", "all_patients"]) & (M.confidence != "low")
    print(f"\n  {len(M)} adjudicated   {int(usable.sum())} usable / "
          f"{M[usable].disease.nunique()} diseases")
    print("\n=== verdict ===")
    print(M.verdict.value_counts().to_string())
    return M


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--packet-dir", default=None,
                    help="directory of packets; built if absent")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--only", nargs="+", metavar="DISEASE__COHORT")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--collect-only", action="store_true",
                    help="gather existing verdicts into the table and stop")
    args = ap.parse_args(argv)

    if args.collect_only:
        collect(args.out_dir)
        return 0

    import glob
    pdir = args.packet_dir or paths.out("packets")
    if not glob.glob(os.path.join(pdir, "*.md")):
        print(f"no packets in {pdir}; building them")
        packets.run()
    files = sorted(glob.glob(os.path.join(pdir, "*.md")))
    if args.only:
        want = set(args.only)
        files = [f for f in files
                 if os.path.basename(f).replace(".md", "") in want]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no packets to adjudicate in {pdir}")
    print(f"{len(files)} cohort packets from {pdir}")

    prompt = open(PROMPT, encoding="utf-8").read()

    def parts(path):
        base = os.path.basename(path).replace(".md", "")
        disease, _, cohort = base.rpartition("__")
        return disease, cohort

    llm.run_batch(
        files,
        key_of=lambda f: os.path.basename(f).replace(".md", ""),
        prompt_of=lambda f: (prompt + "\n\n" + f + "\n\n"
                             "--- packet contents follow ---\n\n"
                             + open(f, encoding="utf-8").read()),
        validate_of=lambda f: validator(*parts(f)),
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

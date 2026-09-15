"""Stage 1: the axis agent.

Public cohorts record whatever their study happened to record, under whatever
names the submitters chose. One writes `sex`, another `Sex`, another
`gender: F`. One records tumour stage, another records the same thing as
`pathologic_T` and `pathologic_N` separately, a third as a free-text histology
string. Deciding which of these can divide patients into comparable groups —
and which would divide them by something read off their own expression, or by
something that had not happened yet when the tissue was taken — is a judgement
about what a variable means, made once per cohort.

That judgement used to be four regular expressions and a hand-written list.
This replaces them.

What the agent does not do is apply its own decision. It returns level
definitions naming the raw values that belong to each, and the split is carried
out mechanically afterwards, so what defines a patient group is written down
and can be checked against the samples it produced.

The preprocessing agent already reads sample characteristics to find the trait,
age and sex of each cohort. This asks the same kind of question of every
variable rather than of three.
"""
import json
import os

import pandas as pd

from grn import paths
from grn.agents import llm
from grn.tools.curation import characteristics as ch

PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "prompts", "axes.md")
OUT_DIR = os.path.join(paths.CURATION_OUT, "axes")

REASONS = {"expression_derived", "post_sampling", "technical",
           "no_contrast", "sample_property", "too_few"}


def validator(packet, floor):
    given = {v["variable"] for v in packet["variables"]}

    def check(obj):
        llm.check_keys(obj, ["disease", "cohort", "disease_group",
                             "axes", "rejected"])
        if str(obj["cohort"]) != str(packet["cohort"]):
            raise llm.LLMError(f"cohort should be {packet['cohort']!r}")
        llm.check_keys(obj["disease_group"], ["variable", "values", "note"])

        seen = []
        for a in obj["axes"]:
            llm.check_keys(a, ["axis", "variable", "kind", "levels", "why"])
            if a["variable"] not in given:
                raise llm.LLMError(
                    f"axes names a variable that was not recorded: {a['variable']}")
            seen.append(a["variable"])
            if a["kind"] not in ("categorical", "ordinal", "numeric_binned"):
                raise llm.LLMError(f"kind must be categorical, ordinal or "
                                   f"numeric_binned, got {a['kind']!r}")
            if len(a["levels"]) < 2:
                raise llm.LLMError(
                    f"axis {a['axis']!r} has fewer than two levels; a variable "
                    f"that cannot divide the patients belongs in rejected")
            for lv in a["levels"]:
                llm.check_keys(lv, ["level", "values", "bounds", "n"])
                if a["kind"] != "numeric_binned" and not lv["values"]:
                    raise llm.LLMError(
                        f"level {lv['level']!r} of {a['axis']!r} lists no raw "
                        f"values, so it cannot be applied to the samples")
        for r in obj["rejected"]:
            llm.check_keys(r, ["variable", "reason"])
            if r["variable"] not in given:
                raise llm.LLMError(
                    f"rejected names a variable that was not recorded: "
                    f"{r['variable']}")
            if r["reason"] not in REASONS:
                raise llm.LLMError(f"reason must be one of {sorted(REASONS)}")
            seen.append(r["variable"])

        missing = given - set(seen)
        if missing:
            raise llm.LLMError(
                "every recorded variable must appear in axes or rejected; "
                f"missing: {', '.join(sorted(missing)[:6])}")
        obj["disease"] = packet["disease"]
        return obj
    return check


def collect(out_dir=OUT_DIR, floor=30):
    """Gather the per-cohort decisions into the two tables the pipeline reads."""
    import glob
    ax_rows, rj_rows = [], []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        if os.path.basename(p) == "failures.jsonl":
            continue
        d = json.load(open(p, encoding="utf-8"))
        for a in d["axes"]:
            for lv in a["levels"]:
                # No floor here. The agent's counts are over the whole cohort,
                # and the group that actually results is the disease group cut
                # by this level, which is only known once the rule is applied.
                # Unit building applies the floor to the real counts.
                ax_rows.append({
                    "disease": d["disease"], "cohort": d["cohort"],
                    "axis": a["axis"], "variable": a["variable"],
                    "kind": a["kind"], "level": lv["level"],
                    "values": "|".join(map(str, lv["values"])),
                    "bounds": lv.get("bounds", ""), "n_expected": int(lv["n"]),
                    "why": a["why"]})
        for r in d["rejected"]:
            rj_rows.append({"disease": d["disease"], "cohort": d["cohort"],
                            "variable": r["variable"], "reason": r["reason"]})
        g = d["disease_group"]
        if g.get("variable"):
            ax_rows.append({
                "disease": d["disease"], "cohort": d["cohort"],
                "axis": "disease_group", "variable": g["variable"],
                "kind": "categorical", "level": "cases",
                "values": "|".join(map(str, g["values"])), "bounds": "",
                "n_expected": 0, "why": g.get("note", "")})
    if not ax_rows:
        raise SystemExit(f"no axis decisions in {out_dir}")
    A = pd.DataFrame(ax_rows)
    R = pd.DataFrame(rj_rows)
    paths.write_table(A, paths.out("condition_axes.csv"), min_rows=10)
    paths.write_table(R, paths.out("condition_axes_rejected.csv"), min_rows=1)
    real = A[A.axis != "disease_group"]
    print(f"\n  {real.axis.nunique()} distinct axes over "
          f"{real.groupby(['disease', 'cohort']).ngroups} cohorts, "
          f"{len(real)} candidate levels")
    print(f"  {len(R)} variables rejected")
    print("\n=== why variables were rejected ===")
    print(R.reason.value_counts().to_string())
    return A, R


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--floor", type=int, default=30,
                    help="samples a level needs to define a patient group")
    ap.add_argument("--only", nargs="+", metavar="DISEASE__COHORT")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args(argv)

    if args.collect_only:
        collect(args.out_dir, args.floor)
        return 0

    paths.require_raw_geo()
    V = paths.verdicts(usable_only=True)
    items = [(r.disease, r.cohort) for _, r in V.iterrows()]
    if args.only:
        want = set(args.only)
        items = [i for i in items if f"{i[0]}__{i[1]}" in want]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} cohorts admitted for the disease group")

    prompt = open(PROMPT, encoding="utf-8").read()
    packets = {}

    def packet_of(item):
        disease, cohort = item
        if item not in packets:
            packets[item] = ch.variables_for(disease, cohort)
        return packets[item]

    def prompt_of(item):
        p = packet_of(item)
        if p is None:
            raise llm.LLMError("no recorded characteristics for this cohort")
        return (prompt.replace("the floor is given below",
                               f"the floor is {args.floor} samples")
                + f"\n\nA level needs at least {args.floor} samples.\n\n---\n\n"
                + ch.describe(p))

    def validate_of(item):
        return validator(packet_of(item), args.floor)

    ok = [i for i in items if packet_of(i) is not None]
    skipped = len(items) - len(ok)
    if skipped:
        print(f"  {skipped} cohorts have no characteristics recorded and are "
              f"left with their disease group undivided")

    llm.run_batch(ok, key_of=lambda i: f"{i[0]}__{i[1]}",
                  prompt_of=prompt_of, validate_of=validate_of,
                  out_dir=args.out_dir, label="cohort",
                  model=args.model, backend=args.backend, jobs=args.jobs,
                  redo=args.redo)
    print()
    collect(args.out_dir, args.floor)
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

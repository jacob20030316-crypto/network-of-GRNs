"""Stage 2: the network agents.

One per construction route — the curated-scaffold network and the network
inferred without a catalogue. Each drives its route's tools in order and then
decides whether what came out may be used.

The division of labour is deliberate and is the reason this file is short. The
correction ladder, the similarity, the joining rules and the permutation null
apply one rule to all 774 patient groups; that uniformity is the property the
whole design rests on, and a model re-deriving those steps per run would only
add variance. What is not uniform is the judgement at the end. A network can
fail because cohort size predicts similarity, or because a correction removed
the confound and the signal with it, and telling those apart means reading
several numbers against each other rather than applying a threshold to one.

So the agent holds the tools, runs them, and inspects. It does not compute the
statistics, and it cannot alter them.

    python -m grn.agents.network --route scaffold --net cond
    python -m grn.agents.network --route scaffold --net cond --inspect-only
"""
import json
import os

import pandas as pd

from grn import paths, runner
from grn.agents import llm
from grn import pipeline

PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "prompts", "network.md")
OUT_DIR = os.path.join(paths.READOUT, "inspection")

# The stage after which the network exists and can be judged. Everything before
# it builds; everything after it reads off what was built.
INSPECT_AFTER = {"scaffold": "similarity", "denovo": "similarity"}


def tools(route):
    """What this agent can call, and what each call leaves behind."""
    return [{"stage": s.name, "does": s.note,
             "produces": [p.split(":", 1)[1] for p in s.produces]}
            for s in pipeline.PHASES[route]]


# What each route's read-out will claim, so the inspection is held to the
# standard that route is actually used at. The de novo route does not assert
# which diseases are related — that is the scaffold's job — so judging it on
# how well it separates diseases would fail it for not doing something it is
# not used for.
PURPOSE = {
    "scaffold": ("This network carries the disease relationships. Its read-out "
                 "names, for a pair of diseases, the transcription factors "
                 "whose targets agree across the pair's patient groups, and "
                 "asserts that those diseases are related through them."),
    "denovo": ("This network supplies candidate regulatory relationships that "
               "a curated catalogue does not contain. Its read-out names, for "
               "a pair of diseases, the regulons the two independently "
               "inferred networks share. It does not establish which diseases "
               "are related; that comes from the curated-scaffold network. "
               "What it must therefore support is that a shared regulon is a "
               "property of the two diseases rather than of the inference."),
}


def evidence(route, net):
    """The numbers the inspection reads. Nothing here is the agent's opinion."""
    ev = {"route": route, "network": net, "purpose": PURPOSE[route],
          "checks": [], "discrimination": []}

    # The confound checks belong to the scaffold construction. The de novo
    # route fixes the same confound differently — a fixed draw of samples and a
    # fixed regulon count per node — so it has no such table, and showing it
    # the scaffold's would be showing it another network's checks.
    acc = os.path.join(paths.READOUT,
                       f"acceptance_{net}.csv" if route == "scaffold"
                       else "acceptance_denovo.csv")
    if os.path.isfile(acc):
        ev["checks"] = pd.read_csv(acc).to_dict("records")

    gate = os.path.join(paths.READOUT, f"similarity_gate_{net}_core.csv")
    if route == "scaffold" and os.path.isfile(gate):
        G = pd.read_csv(gate)
        keep = [c for c in ("method", "same_disease", "auc",
                            "auc_tissue_matched", "spearman_n", "source_r2")
                if c in G.columns]
        ev["discrimination"] = G[keep].round(4).to_dict("records")

    meta = os.path.join(paths.SCAFFOLD_OUT, f"meta_{net}.csv")
    if route == "scaffold" and os.path.isfile(meta):
        M = pd.read_csv(meta)
        ev["size"] = {"nodes": len(M), "diseases": int(M.disease.nunique()),
                      "cohorts": int(M.groupby(["disease", "cohort"]).ngroups)}

    if route == "denovo":
        cmp_ = os.path.join(paths.DENOVO_OUT, "denovo_method_comparison.csv")
        if os.path.isfile(cmp_):
            ev["discrimination"] = pd.read_csv(cmp_).round(4).to_dict("records")
        import glob
        reg = [p for p in glob.glob(os.path.join(paths.DENOVO_OUT,
                                                 "*.regulons.tsv"))
               if not os.path.basename(p).split(".")[0].endswith(
                   tuple(f"__r{i}" for i in range(10)))]
        if reg:
            ev["size"] = {"nodes": len(reg)}
    return ev


def brief(ev):
    lines = [f"route: {ev['route']}", f"network: {ev['network']}",
             "", "what this network is used for:", f"  {ev['purpose']}", ""]
    if ev.get("size"):
        lines.append("size: " + ", ".join(f"{v} {k}" for k, v in ev["size"].items()))
    lines += ["", "acceptance checks:"]
    for c in ev["checks"]:
        lines.append(f"  {c['check']:<30} {c['value']:>+8.4f}   "
                     f"threshold |{c['threshold']}|   "
                     f"{'pass' if c['passes'] else 'FAIL'}")
    if not ev["checks"]:
        lines.append("  (none available)")
    lines += ["", "discrimination, by similarity definition:"]
    for d in ev["discrimination"]:
        lines.append("  " + "   ".join(f"{k}={v}" for k, v in d.items()))
    if not ev["discrimination"]:
        lines.append("  (none available)")
    return "\n".join(lines)


def validator(ev):
    keys = {c["check"] for c in ev["checks"]}

    def check(obj):
        llm.check_keys(obj, ["network", "proceed", "failed_checks",
                             "marginal_checks", "discrimination",
                             "assessment", "notes"])
        if not isinstance(obj["proceed"], bool):
            raise llm.LLMError("proceed must be true or false")
        for f in ("failed_checks", "marginal_checks"):
            bad = [k for k in obj[f] if k not in keys]
            if bad:
                raise llm.LLMError(
                    f"{f} names checks that were not run: {', '.join(bad)}")
        # A network cannot both fail a check and be cleared: the threshold is
        # the criterion, not a starting point for negotiation.
        if obj["failed_checks"] and obj["proceed"]:
            raise llm.LLMError(
                "proceed cannot be true while failed_checks is non-empty")
        return obj
    return check


def inspect(route, net, model=None, backend=None):
    """Read the checks, judge, write the verdict beside them."""
    ev = evidence(route, net)
    if not ev["checks"] and not ev["discrimination"]:
        raise SystemExit(
            f"nothing to inspect for route {route}: build it first\n"
            f"  python -m grn.agents.network --route {route} --net {net}")
    obj, attempts = llm.ask(
        open(PROMPT, encoding="utf-8").read() + "\n\n---\n\n" + brief(ev),
        validator(ev), model=model, backend=backend)
    obj["evidence"] = ev
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{route}_{net}.json")
    json.dump(obj, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  proceed: {obj['proceed']}")
    if obj["failed_checks"]:
        print(f"  failed:   {', '.join(obj['failed_checks'])}")
    if obj["marginal_checks"]:
        print(f"  marginal: {', '.join(obj['marginal_checks'])}")
    print(f"  {obj['assessment']}")
    print(f"\nwrote {path}" + ("" if attempts == 1 else f"  ({attempts} attempts)"))
    return obj


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--route", choices=sorted(pipeline.PHASES), default="scaffold")
    ap.add_argument("--net", choices=["uncond", "cond"], default="cond")
    ap.add_argument("--inspect-only", action="store_true",
                    help="judge what is already built, run nothing")
    ap.add_argument("--build-only", action="store_true",
                    help="run the tools, skip the judgement")
    ap.add_argument("--list-tools", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args(argv)

    if args.list_tools:
        for t in tools(args.route):
            print(f"  {t['stage']:<14} {t['does']}")
            print(f"                 -> {', '.join(t['produces'])}")
        return 0

    stages = pipeline.PHASES[args.route]
    stop_after = INSPECT_AFTER[args.route]

    if not args.inspect_only:
        problem = pipeline.precheck(args.route)
        if problem:
            print(f"\ncannot start route '{args.route}':\n  {problem}\n")
            return 1

        # Build up to the point where the network exists, inspect it, and only
        # then read anything off it. Running the read-out first and judging
        # afterwards would make the judgement decorative.
        names = [s.name for s in stages]
        upto = stages[:names.index(stop_after) + 1]
        runner.run_route(upto, args.route, args.net, force=args.force)

        if not args.build_only:
            verdict = inspect(args.route, args.net, args.model, args.backend)
            if not verdict["proceed"]:
                print("\nthe network did not pass inspection; the read-out is "
                      "not run.\n  to override, build the remaining stages with "
                      f"run.py build --route {args.route} --from "
                      f"{names[names.index(stop_after) + 1]}")
                return 1

        rest = stages[names.index(stop_after) + 1:]
        if rest:
            runner.run_route(rest, args.route, args.net, force=args.force)
        return 0

    inspect(args.route, args.net, args.model, args.backend)
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

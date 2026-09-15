"""Stage 3, second half: the biologist agent.

It reads one pair's evidence pack and writes the mechanistic account, with a
verdict on what the relationship is and a separate judgement of how far the
evidence carries it. Those two are kept apart on purpose: a well-supported
account of a known axis and a thinly-supported guess at a new one are different
results, and one field cannot say both.

What it is shown, and what it is not:

  shown       the factors, how many of the pair's patient groups each recurred
              in, the direction of coupling, the targets, and the retrieved
              record with the assistant's reading of it. Also the two lists of
              factors that did not survive.

  withheld    every p-value, q-value and rank. The account has to stand on the
              biology of the list; if the agent could see how strongly a pair
              scored, agreement between its reading and our statistics would
              stop being evidence and start being an echo.

The exclusions are shown rather than hidden because they are what the selection
criterion buys. A pair's story is not "these two diseases share X"; it is
"they share X rather than Y, though Y led an individual patient group". Without
the second half a reader cannot see what recurrence bought, and the agent is
liable to reach for exactly the tissue master regulators that were removed.
"""
import json
import os
import re

from grn import paths
from grn.agents import llm

PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "prompts", "biologist.md")
EVIDENCE_DIR = os.path.join(paths.AGENT_OUT, "evidence")
OUT_DIR = os.path.join(paths.AGENT_OUT, "interpretations")


def _factor_block(e, rec, a, b):
    """One factor: what we measured, then what the record holds."""
    out = [f"\n  {rec['tf']}   {rec['direction']}, "
           f"supported in {round(rec['recurrence'] * rec.get('_n', 0))} / "
           f"{rec.get('_n', 0)} patient-group comparisons, "
           f"{rec['n_targets']} targets"]
    tg = rec.get("targets", [])
    if tg:
        out.append(f"    targets: {', '.join(tg[:24])}"
                   + (" …" if len(tg) > 24 else ""))
    if e is None:
        return out
    for d in (a, b):
        ot = e["open_targets"].get(d, {})
        if ot.get("known"):
            out.append(f"    Open Targets in {d}: percentile {ot['percentile']} "
                       f"of {ot['n_scored_for_disease']} scored genes")
        else:
            out.append(f"    Open Targets in {d}: no association recorded")
    chip = e.get("chip", {})
    if chip.get("in_library"):
        out.append(f"    observed bound at {chip['n_bound']} of "
                   f"{chip['n_targets']} of its targets (ReMap)")
    seen = e.get("assessed")
    lit = e.get("literature", {})
    papers = {}
    for v in lit.values():
        for p in v.get("papers", []):
            papers[str(p.get("pmid"))] = p
    for v in e.get("literature_extra", []) or []:
        for p in v.get("results", []):
            papers[str(p.get("pmid"))] = p
    if seen:
        for field, label in (("supported_in_a", f"places it in {a}"),
                             ("supported_in_b", f"places it in {b}"),
                             ("joint", "concerns both diseases")):
            for pmid in seen.get(field, []):
                p = papers.get(str(pmid), {})
                out.append(f"    PMID {pmid} — {label}: "
                           f"{str(p.get('title', ''))[:100]}")
        n_unc = len(seen.get("uncertain", []))
        if n_unc:
            out.append(f"    {n_unc} further retrieved paper(s) could not be "
                       f"judged from the text available")
        if seen.get("note"):
            out.append(f"    reader's note: {seen['note']}")
    elif not papers:
        out.append("    nothing retrieved for this factor")
    return out


def brief(pack):
    a, b = pack["disease_a"], pack["disease_b"]
    n = pack.get("n_views", 0)
    lines = [f"pair_id: {pack['pair']}",
             f"disease A: {a}",
             f"disease B: {b}",
             f"tissue pairing: {pack.get('tissue_pairing', 'unknown')}",
             f"patient-group comparisons available: {n}",
             f"independent cohort pairs: {pack.get('n_cohort_pairs', '?')}",
             "", "factors:"]
    by_tf = {e["tf"].upper(): e for e in pack.get("evidence", [])}
    for rec in pack["tfs"]:
        rec = dict(rec, _n=n)
        lines += _factor_block(by_tf.get(rec["tf"].upper()), rec, a, b)

    exc = pack.get("excluded", {})
    sv = exc.get("single_view_would_report", [])
    if sv:
        lines += ["", "did not survive — led an individual patient group only:"]
        for x in sv[:12]:
            lines.append(f"  {x['tf']}  top of one group at percentile "
                         f"{x.get('best_single_view_percentile')}, "
                         f"recurrence across all groups "
                         f"{x.get('recurrence_across_all_views')}")
    tp = exc.get("as_tissue_programme", [])
    if tp:
        lines += ["", "removed as generic to these tissues:",
                  "  " + ", ".join(str(x) for x in tp[:24])]
    cov = (pack.get("assistant") or {}).get("coverage")
    if cov:
        lines += ["", f"on the record overall: {cov}"]
    return "\n".join(lines)


def validator(pack):
    factors = [t["tf"] for t in pack["tfs"]]
    pmids = set()
    for e in pack.get("evidence", []):
        for v in e.get("literature", {}).values():
            pmids |= {str(p["pmid"]) for p in v.get("papers", []) if p.get("pmid")}
        for v in e.get("literature_extra", []) or []:
            pmids |= {str(p["pmid"]) for p in v.get("results", []) if p.get("pmid")}

    VERDICTS = {"established", "supported", "unexpected", "incoherent"}
    STRENGTH = {"strong", "moderate", "weak", "insufficient"}

    def check(obj):
        llm.check_keys(obj, ["pair_id", "programme", "carrying_factors",
                             "role_in_a", "role_in_b", "shared_axis", "verdict",
                             "support_strength", "reasoning", "support",
                             "how_to_falsify", "caveats"])
        if obj["pair_id"] != pack["pair"]:
            raise llm.LLMError(f"pair_id should be {pack['pair']!r}")
        if obj["verdict"] not in VERDICTS:
            raise llm.LLMError(f"verdict must be one of {sorted(VERDICTS)}")
        if obj["support_strength"] not in STRENGTH:
            raise llm.LLMError(f"support_strength must be one of {sorted(STRENGTH)}")
        llm.check_vocabulary(obj, "carrying_factors", factors, "factor")
        llm.check_citations(obj, pmids, fields=("support", "reasoning",
                                                "shared_axis"))
        if obj["verdict"] != "incoherent" and not obj["carrying_factors"]:
            raise llm.LLMError("a verdict other than 'incoherent' needs at least "
                               "one carrying factor")
        if not str(obj["how_to_falsify"]).strip():
            raise llm.LLMError("how_to_falsify is required; an account without "
                               "one is not a hypothesis")
        return obj
    return check


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evidence-dir", default=EVIDENCE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args(argv)

    import glob
    packs = []
    for p in sorted(glob.glob(os.path.join(args.evidence_dir, "*.json"))):
        if os.path.basename(p) == "failures.jsonl":
            continue
        packs.append(json.load(open(p, encoding="utf-8")))
    if args.only:
        packs = [p for p in packs if p["pair"] in set(args.only)]
    if args.limit:
        packs = packs[:args.limit]
    if not packs:
        raise SystemExit(
            f"no evidence packs in {args.evidence_dir}\n"
            f"  the assistant agent builds them:\n"
            f"      python -m grn.agents.assistant --backend openai")
    print(f"{len(packs)} evidence packs from "
          f"{os.path.basename(args.evidence_dir)}")

    prompt = open(PROMPT, encoding="utf-8").read()
    return llm.run_batch(
        packs,
        key_of=lambda p: re.sub(r"[^A-Za-z0-9_.-]", "_",
                                p["pair"].replace("||", "__")),
        prompt_of=lambda p: prompt + "\n\n---\n\n" + brief(p),
        validate_of=validator,
        out_dir=args.out_dir, label="pair",
        model=args.model, backend=args.backend, jobs=args.jobs,
        redo=args.redo)


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

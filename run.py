#!/usr/bin/env python
"""The entry point.

    python run.py plan                      the whole pipeline, and what is done
    python run.py paths                     where every input and output resolved
    python run.py all                       run it, start to finish
    python run.py curate                    one phase
    python run.py scaffold --net cond
    python run.py scaffold --only mechanism
    python run.py denovo --from pairs
    python run.py interpret

Two steps are outside this file because they should not be restarted casually:

    cd preprocess && python main.py --data-root <raw> --version v1 \\
        --model gpt-4.1 --skip-filter          cohorts -> expression matrices
    nohup python scripts/infer_denovo.py --repeats 3 &        two days

Each phase checks that what it needs is present and says how to produce it if
not. A stage that has run and whose parameters have not changed is skipped, so
a rerun continues rather than restarts.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grn import paths, runner                                    # noqa: E402
from grn.pipeline import ORDER, PHASES, precheck                 # noqa: E402


def cmd_plan(args):
    print("\nthe pipeline\n")
    for phase in ORDER:
        stages = PHASES[phase]
        print(f"  {phase}")
        for s in stages:
            mark = "agent" if s.agent else "     "
            print(f"    {mark}  {s.name:<12} {s.note}")
        print()
    print("  preprocess/ and scripts/infer_denovo.py run separately; "
          "see the module docstring.\n")
    return 0


def cmd_paths(args):
    print(f"PROGRAM = {paths.PROGRAM}")
    print(f"DATA    = {paths.DATA}\n")
    paths.describe()
    problems = paths.check()
    if problems:
        print("\nnot yet present:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\ninputs complete")
    return 0


def _run_phase(phase, args):
    problem = precheck(phase)
    if problem:
        print(f"\ncannot start '{phase}':\n  {problem}\n")
        return 1
    net = getattr(args, "net", "cond")
    ran = runner.run_route(
        PHASES[phase], phase, net,
        only=getattr(args, "only", None),
        start_from=getattr(args, "from", None),
        force=getattr(args, "force", False),
        dry_run=getattr(args, "dry_run", False),
        overrides={},
    )
    if not getattr(args, "dry_run", False):
        print(f"{phase}: {len(ran)} stage(s) ran"
              + (f" — {', '.join(ran)}" if ran else ""))
    return 0


def cmd_phase(args):
    return _run_phase(args.cmd, args)


def cmd_all(args):
    for phase in ORDER:
        rc = _run_phase(phase, args)
        if rc:
            print(f"\nstopped at '{phase}'.")
            return rc
    print("\nthe pipeline is complete.")
    return 0


def cmd_status(args):
    for phase in ORDER:
        runner.status(PHASES[phase], phase, getattr(args, "net", "cond"))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("plan", help="the whole pipeline, and which steps are agents")
    sub.add_parser("paths", help="where every input and output resolved to")

    def add_run_args(p, net=True):
        if net:
            p.add_argument("--net", choices=["uncond", "cond"], default="cond")
        p.add_argument("--only", nargs="+", metavar="STAGE")
        p.add_argument("--from", metavar="STAGE")
        p.add_argument("--force", action="store_true",
                       help="rerun even if up to date")
        p.add_argument("--dry-run", action="store_true")

    add_run_args(sub.add_parser("all", help="run every phase in order"))
    add_run_args(sub.add_parser("status", help="what has been built"))
    for phase in ORDER:
        add_run_args(sub.add_parser(phase, help=f"run the {phase} phase"))

    args = ap.parse_args()
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "paths":
        return cmd_paths(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "all":
        return cmd_all(args)
    return cmd_phase(args)


if __name__ == "__main__":
    sys.exit(main())

"""Stage machinery: declare a route once, run it, resume it, compare it.

A route is an ordered list of Stages. A Stage knows three things — what to
call, what files that call must leave behind, and whether it runs once or once
per network. Nothing else about the science lives here; every stage body is the
ported analysis script, unchanged.

Resuming is by artefact plus parameters. A stage is skipped when every file it
promises exists and the parameters it would run with hash to the same value as
the recorded run. Changing a parameter therefore re-runs that stage without
anyone having to remember to pass --force.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from grn import paths

# Where a promised artefact lives. Stages name their outputs as "dir:file".
DIRS = {
    "curation": paths.CURATION_OUT,
    "scaffold": paths.SCAFFOLD_OUT,
    "denovo": paths.DENOVO_OUT,
    "readout": paths.READOUT,
    "agent": paths.AGENT_OUT,
}


@dataclass
class Stage:
    name: str
    fn: Callable
    produces: Sequence[str]
    per_net: bool = True
    params: dict = field(default_factory=dict)
    note: str = ""
    net_kw: str = "net"          # some ported scripts spell the flag --network
    agent: bool = False          # a step where a model decides, not a calculation

    def outputs(self, net):
        out = []
        for spec in self.produces:
            where, fname = spec.split(":", 1)
            out.append(os.path.join(DIRS[where], fname.format(net=net)))
        return out

    def call_params(self, net, overrides):
        p = dict(self.params)
        p.update({k: v for k, v in overrides.items() if v is not None})
        if self.per_net:
            p.setdefault(self.net_kw, net)
        # a parameter may name a file belonging to one network
        return {k: (v.format(net=net) if isinstance(v, str) and "{net}" in v else v)
                for k, v in p.items()}


def _fingerprint(params):
    return hashlib.sha1(
        json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _mark_path(route, stage, net):
    tag = f"{route}.{net}.{stage}" if net else f"{route}.{stage}"
    return os.path.join(paths.DONE, tag + ".json")


def _is_done(stage, route, net, params):
    mark = _mark_path(route, stage.name, net if stage.per_net else None)
    if not os.path.isfile(mark):
        return False, "never run"
    try:
        rec = json.load(open(mark, encoding="utf-8"))
    except Exception:
        return False, "unreadable checkpoint"
    if rec.get("fingerprint") != _fingerprint(params):
        return False, "parameters changed"
    missing = [p for p in stage.outputs(net) if not os.path.exists(p)]
    if missing:
        return False, f"missing {os.path.basename(missing[0])}"
    return True, "up to date"


def _record(stage, route, net, params, seconds):
    mark = _mark_path(route, stage.name, net if stage.per_net else None)
    outs = stage.outputs(net)
    json.dump({
        "route": route,
        "stage": stage.name,
        "net": net if stage.per_net else None,
        "params": {k: str(v) for k, v in params.items()},
        "fingerprint": _fingerprint(params),
        "seconds": round(seconds, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outputs": [{"path": p, "bytes": os.path.getsize(p)}
                    for p in outs if os.path.exists(p)],
    }, open(mark, "w", encoding="utf-8"), indent=2)


def _invalidate_downstream(stages, done_stage, route, net):
    """A stage that just ran makes every later stage stale.

    Stages are a chain: each reads what the one before it wrote. Without this,
    re-running an early stage with a new parameter leaves the later ones
    marked up to date, and the route silently mixes products from two
    different settings — which is exactly the kind of inconsistency the
    checkpoints exist to prevent.
    """
    after = stages[[s.name for s in stages].index(done_stage.name) + 1:]
    for s in after:
        mark = _mark_path(route, s.name, net if s.per_net else None)
        if os.path.isfile(mark):
            os.remove(mark)
            print(f"        (invalidated '{s.name}' — it reads what this stage wrote)")


def run_route(stages, route, net, only=None, start_from=None,
              force=False, overrides=None, dry_run=False):
    """Execute one route. Returns the list of stages that actually ran."""
    overrides = overrides or {}
    if only:
        wanted = [s for s in stages if s.name in only]
        unknown = set(only) - {s.name for s in stages}
        if unknown:
            raise SystemExit(f"no such stage: {', '.join(sorted(unknown))}")
    elif start_from:
        names = [s.name for s in stages]
        if start_from not in names:
            raise SystemExit(f"no such stage: {start_from}")
        wanted = stages[names.index(start_from):]
    else:
        wanted = list(stages)

    print(f"\nroute {route}" + (f"  net {net}" if net else "")
          + f"   {len(wanted)} of {len(stages)} stages")
    print("=" * 72)

    ran = []
    for s in wanted:
        params = s.call_params(net, overrides)
        done, why = _is_done(s, route, net, params)
        label = f"{s.name:<18}"
        if done and not force:
            print(f"  skip  {label} {why}")
            continue
        if dry_run:
            shown = " ".join(f"{k}={v}" for k, v in params.items())
            print(f"  would run {label} {shown}")
            print(f"        -> {', '.join(os.path.basename(p) for p in s.outputs(net))}")
            continue
        print(f"\n  run   {label} {why if not force else 'forced'}")
        if s.note:
            print(f"        {s.note}")
        print("  " + "-" * 68)
        t0 = time.time()
        s.fn(**params)
        dt = time.time() - t0
        missing = [p for p in s.outputs(net) if not os.path.exists(p)]
        if missing:
            raise SystemExit(
                f"\nstage '{s.name}' finished but did not write:\n  "
                + "\n  ".join(missing)
                + "\n(the stage's declared outputs are wrong, or it failed quietly)")
        _record(s, route, net, params, dt)
        _invalidate_downstream(stages, s, route, net)
        print(f"  " + "-" * 68)
        print(f"  done  {label} {dt:.1f}s -> "
              f"{', '.join(os.path.basename(p) for p in s.outputs(net))}")
        ran.append(s.name)
    print()
    return ran


def status(stages, route, net, overrides=None):
    overrides = overrides or {}
    print(f"\nroute {route}" + (f"  net {net}" if net else ""))
    print(f"  {'stage':<18} {'state':<22} outputs")
    print("  " + "-" * 70)
    for s in stages:
        params = s.call_params(net, overrides)
        done, why = _is_done(s, route, net, params)
        mark = "ok" if done else "--"
        files = ", ".join(os.path.basename(p) for p in s.outputs(net))
        print(f"  {s.name:<18} {mark} {why:<19} {files}")
    print()

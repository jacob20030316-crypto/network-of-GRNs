"""One place where a model is called, and every way that can go wrong.

Agents in this pipeline all do the same shape of work: read one item, return one
JSON object with a fixed set of keys. What differs is the prompt. So the model
call, the parsing, the checking and the recovery live here once, and an agent is
then a prompt plus a schema.

Failures are expected rather than exceptional, and each has its own answer:

  the call fails                 retry with backoff, then record the item as
                                 failed and carry on. One bad item must never
                                 take down a batch of two hundred.
  the reply is not JSON          pull the outermost object out of whatever came
                                 back; models wrap JSON in prose and fences.
  the JSON is missing keys       hand the error back and let it try again; a
                                 second failure is recorded, not patched.
  the reply invents a factor     rejected. An agent may only name factors that
                                 were given to it.
  the reply invents a citation   rejected. An agent may only cite identifiers
                                 that appear in its evidence pack. This is the
                                 one failure a reader cannot detect for
                                 themselves, so it is checked mechanically
                                 rather than discouraged in the prompt.
  the batch dies halfway         every item is written to its own file and an
                                 item that already has one is skipped, so a
                                 rerun resumes instead of restarting.

Everything rejected is kept in `failures.jsonl` with the reason and the raw
reply. A batch reports what it could not do rather than quietly returning less.
"""
import json
import os
import re
import subprocess
import time

BACKEND = os.environ.get("GRN_LLM", "claude-cli")
MODEL = os.environ.get("GRN_LLM_MODEL", "sonnet")
RETRIES = int(os.environ.get("GRN_LLM_RETRIES", "3"))


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

def _call_claude_cli(prompt, model, timeout):
    """Headless Claude Code. No API key: it uses the local session's credentials.

    This is the backend the cohort adjudication already ran on, so it is known
    to survive a few hundred sequential calls.
    """
    p = subprocess.run(["claude", "-p", "--model", model],
                       input=prompt, capture_output=True, text=True,
                       timeout=timeout)
    if p.returncode != 0:
        raise LLMError(f"claude exited {p.returncode}: {p.stderr.strip()[:300]}")
    return p.stdout


def _key(name):
    """An API key from the environment, without whatever the file it came from
    left on the end. A key sourced from a CRLF `.env` carries a carriage return,
    which becomes an illegal HTTP header and fails as a connection error —
    several layers away from the actual cause."""
    v = os.environ.get(name, "").strip()
    if not v:
        raise LLMError(f"{name} is not set")
    return v


def _call_openai(prompt, model, timeout):
    """Any OpenAI-compatible endpoint, for running this outside a Claude session.

    Plain HTTP rather than the vendor SDK. Chat completions is one POST, the
    project already depends on `requests` for Europe PMC, and an SDK is one
    more thing that can be incompatible with whatever else is installed —
    which is exactly what happened here. Setting OPENAI_BASE_URL points this at
    any compatible endpoint.
    """
    import requests
    base = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    r = requests.post(
        f"{base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {_key('OPENAI_API_KEY')}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=timeout)
    if r.status_code != 200:
        raise LLMError(f"openai returned {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _call_anthropic(prompt, model, timeout):
    """The Anthropic API, for anyone holding a key rather than a subscription."""
    import anthropic
    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY"))
    r = client.messages.create(
        model=model, max_tokens=8192, timeout=timeout,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


BACKENDS = {"claude-cli": _call_claude_cli,
            "openai": _call_openai,
            "anthropic": _call_anthropic}

# What each backend is asked for when no model is named. The two roles of
# Stage 3 are run on different backends deliberately — see agents/README.
DEFAULT_MODEL = {"claude-cli": "sonnet",
                 "openai": "gpt-4.1",
                 "anthropic": "claude-sonnet-4-5"}


def call(prompt, model=None, backend=None, timeout=600):
    backend = backend or BACKEND
    if backend not in BACKENDS:
        raise LLMError(f"unknown backend {backend!r}; have {sorted(BACKENDS)}")
    model = model or os.environ.get("GRN_LLM_MODEL") or DEFAULT_MODEL[backend]
    return BACKENDS[backend](prompt, model, timeout)


# --------------------------------------------------------------------------
# getting an object out of a reply
# --------------------------------------------------------------------------

def extract_json(text):
    """The outermost JSON object in a reply, however it was wrapped."""
    if text is None:
        raise LLMError("empty reply")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start < 0:
        raise LLMError(f"no JSON object in reply: {text.strip()[:200]}")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as e:
                    raise LLMError(f"malformed JSON: {e}")
    raise LLMError("unterminated JSON object in reply")


# --------------------------------------------------------------------------
# checking it
# --------------------------------------------------------------------------

def check_keys(obj, required):
    missing = [k for k in required if k not in obj]
    if missing:
        raise LLMError(f"missing required keys: {', '.join(missing)}")
    return obj


def check_vocabulary(obj, field, allowed, label="value"):
    """A field may only draw on terms that were supplied.

    Used for the factor names an account is allowed to build on, and for the
    identifiers it is allowed to cite. Both are cases where a plausible
    invention is indistinguishable from a real answer to a reader.
    """
    got = obj.get(field)
    got = got if isinstance(got, list) else ([got] if got else [])
    allowed = {str(a).upper() for a in allowed}
    bad = [g for g in got if str(g).upper() not in allowed]
    if bad:
        raise LLMError(f"{field} names {label}s that were not supplied: "
                       f"{', '.join(map(str, bad[:6]))}")
    return obj


def check_citations(obj, allowed_pmids, fields=("support",)):
    """Every identifier anywhere in these fields must come from the pack."""
    allowed = {str(p) for p in allowed_pmids if p}
    for f in fields:
        text = json.dumps(obj.get(f, ""), ensure_ascii=False)
        cited = set(re.findall(r"\b(?:PMID[:\s]*)?(\d{7,8})\b", text))
        invented = cited - allowed
        if invented:
            raise LLMError(
                f"{f} cites identifiers absent from the evidence pack: "
                f"{', '.join(sorted(invented)[:6])}")
    return obj


# --------------------------------------------------------------------------
# one item, with recovery
# --------------------------------------------------------------------------

def ask(prompt, validate, model=None, backend=None, retries=None, timeout=600,
        on_retry=None):
    """Call, parse, validate; on failure hand the reason back and try again.

    `validate` receives the parsed object and either returns it (possibly
    normalised) or raises LLMError describing what is wrong. That description
    is what the model is shown on the next attempt, so a recoverable mistake
    usually is recovered.
    """
    retries = RETRIES if retries is None else retries
    attempt, last, text = 0, None, None
    while attempt < retries:
        attempt += 1
        try:
            ask_this = prompt if last is None else (
                f"{prompt}\n\n---\nYour previous reply was rejected:\n"
                f"  {last}\nReturn the corrected JSON object and nothing else.")
            text = call(ask_this, model=model, backend=backend, timeout=timeout)
            return validate(extract_json(text)), attempt
        except LLMError as e:
            last = str(e)
        except subprocess.TimeoutExpired:
            last = f"no reply within {timeout}s"
        except Exception as e:                            # network, quota, …
            last = f"{type(e).__name__}: {e}"
        if on_retry:
            on_retry(attempt, last)
        if attempt < retries:
            time.sleep(min(2 ** attempt, 30))
    raise LLMError(f"gave up after {retries} attempts; last: {last}\n"
                   f"raw reply: {(text or '')[:400]}")


def run_batch(items, key_of, prompt_of, validate_of, out_dir, label="item",
              model=None, backend=None, limit=None, redo=False, jobs=1):
    """One file per item, skip what is already done, never abort on one failure.

    Items are independent, so they run concurrently; a model call is minutes of
    waiting and almost no local work. `jobs` is how many at once. Ordering of
    the console output is therefore completion order, not input order.

    Returns (n_written, n_skipped, failures). Failures are also appended to
    `failures.jsonl` beside the outputs, with the reason, so a run can be
    audited without reading the console.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    os.makedirs(out_dir, exist_ok=True)
    fail_path = os.path.join(out_dir, "failures.jsonl")
    lock = threading.Lock()
    failures = []
    todo, skipped = [], 0
    for item in (list(items)[:limit] if limit else list(items)):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key_of(item))
        path = os.path.join(out_dir, f"{safe}.json")
        if os.path.exists(path) and not redo:
            skipped += 1
        else:
            todo.append((item, path))

    def one(pair):
        item, path = pair
        key = key_of(item)
        t0 = time.time()
        try:
            obj, attempts = ask(prompt_of(item), validate_of(item),
                                model=model, backend=backend)
        except LLMError as e:
            with lock:
                failures.append({"key": key, "reason": str(e)})
                with open(fail_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"key": key, "reason": str(e)},
                                        ensure_ascii=False) + "\n")
            return key, None, time.time() - t0, str(e)
        json.dump(obj, open(path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        return key, attempts, time.time() - t0, None

    written = 0
    print(f"{len(todo)} to do, {skipped} already present, {jobs} at a time")
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futures = [ex.submit(one, p) for p in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            key, attempts, dt, err = fut.result()
            if err:
                print(f"  [{i}/{len(todo)}] {label} {key[:46]:<46} FAILED  "
                      f"{err.splitlines()[0][:60]}", flush=True)
            else:
                written += 1
                note = "" if attempts == 1 else f"  ({attempts} attempts)"
                print(f"  [{i}/{len(todo)}] {label} {key[:46]:<46} "
                      f"{dt:5.1f}s{note}", flush=True)
    print(f"\n{written} written, {skipped} already present, "
          f"{len(failures)} failed")
    if failures:
        print(f"  failures listed in {fail_path}")
    return written, skipped, failures

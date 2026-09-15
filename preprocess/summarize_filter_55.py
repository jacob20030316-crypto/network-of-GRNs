"""Parse the filter run log and emit a per-disease summary table.

Outputs:
  - CSV: <out_root>/_summary.csv  (one row per disease, all subset counts)
  - stdout: human-readable summary with breakdowns
"""
import csv
import os
import re
import sys
from collections import defaultdict


LOG_PATH = "./output/filter_55/_run.log"
CSV_OUT = "./output/filter_55/_summary.csv"

DISEASE_HDR = re.compile(r"^\[(\d+)/\d+\]\s+(.+)$")
TRYING = re.compile(r":\s+trying\s+(\S+?)\.csv\s+\(trait_is_control=(True|False)\)")
SUBSET = re.compile(r":\s+(\S+?)\.csv\s+->\s+(\d+)\s+samples")
COMPLETED = re.compile(r":\s+completed using\s+(\S+?)\.csv")


def classify_subset(filename_stem, disease):
    """Categorize a subset-file stem and pull out the trait label when present.

    Handles two filename formats:
      - Old / control / TCGA: {cohort}_{disease}_{val}            (val = 0 or 1)
      - New non-control:      {cohort}_{disease}_{val}_{label}    (val = 0 or 1)

    Returns (category, label_or_None) where category is one of trait_0 /
    trait_1 / age_* / gender_* / other."""
    s = filename_stem
    if s.endswith("_Age_under40"):
        return "age_under40", None
    if s.endswith("_Age_40-60"):
        return "age_40_60", None
    if s.endswith("_Age_over60"):
        return "age_over60", None
    if s.endswith("_Gender_1"):
        return "gender_1", None
    if s.endswith("_Gender_0"):
        return "gender_0", None
    m = re.search(rf"_{re.escape(disease)}_([01])(?:_(.+))?$", s)
    if m:
        return f"trait_{m.group(1)}", m.group(2)
    return "other", None


def _new_row(disease):
    return {
        "disease": disease,
        "cohort_id": "",
        "trait_is_control": "",
        "status": "unknown",
        "trait_1": "",
        "trait_0": "",
        "label_0": "",
        "label_1": "",
        "age_under40": "",
        "age_40_60": "",
        "age_over60": "",
        "gender_1": "",
        "gender_0": "",
        "files_saved": 0,
    }


def parse_log(log_path):
    """Parse the log into one row per disease, keeping the LAST occurrence so
    that a rerun appended to the log overrides the earlier entry."""
    by_disease = {}
    cur = None

    with open(log_path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = DISEASE_HDR.match(line)
            if m:
                # Start a fresh row for this occurrence (overrides any prior one)
                cur = _new_row(m.group(2))
                by_disease[m.group(2)] = cur
                continue
            if cur is None:
                continue

            m = TRYING.search(line)
            if m:
                cur["cohort_id"] = m.group(1)
                cur["trait_is_control"] = m.group(2)
                continue

            m = SUBSET.search(line)
            if m:
                stem = m.group(1)
                n = int(m.group(2))
                cat, label = classify_subset(stem, cur["disease"])
                if cat in cur and cur[cat] == "":
                    cur[cat] = n
                if cat == "trait_0" and label and not cur["label_0"]:
                    cur["label_0"] = label
                if cat == "trait_1" and label and not cur["label_1"]:
                    cur["label_1"] = label
                if n >= 10:
                    cur["files_saved"] += 1
                if stem.startswith("TCGA_") and cur["cohort_id"] == "":
                    cur["cohort_id"] = "TCGA"
                    cur["trait_is_control"] = "True"
                continue

            m = COMPLETED.search(line)
            if m:
                cur["status"] = "completed"
                continue

            # Only apply status changes when the line explicitly references
            # the current disease — guards against stray WARNINGs (e.g. from
            # buffered stdout in tee'd reruns) being attributed to whichever
            # disease block was last opened.
            if cur["disease"] not in line:
                continue
            if "no CSV files" in line:
                cur["status"] = "no_input_data"
            elif "no trait column detected" in line and cur["status"] == "unknown":
                cur["status"] = "no_trait_column"
            elif "exhausted all GEO cohorts" in line:
                cur["status"] = "fallback_failed"
            elif "TCGA dataset has no usable trait column" in line:
                cur["status"] = "no_trait_column"
            elif "TCGA Disease=1 subset has only" in line:
                cur["status"] = "tcga_low_disease_count"

    rows = list(by_disease.values())
    for r in rows:
        if r["status"] == "unknown" and r["cohort_id"] == "TCGA":
            r["status"] = "completed"
    return rows


def write_csv(rows, csv_path):
    fields = ["disease", "cohort_id", "trait_is_control", "status",
              "trait_1", "trait_0", "label_1", "label_0",
              "age_under40", "age_40_60", "age_over60",
              "gender_1", "gender_0",
              "files_saved"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_summary(rows):
    n = len(rows)
    by_status = defaultdict(int)
    by_path = defaultdict(int)
    no_files = []
    only_control = []  # has trait_0 but no trait_1
    only_disease = []  # has trait_1 but no trait_0

    for r in rows:
        by_status[r["status"]] += 1

        if r["cohort_id"] == "TCGA":
            by_path["TCGA"] += 1
        elif r["cohort_id"].startswith("GSE"):
            key = "GEO_control" if r["trait_is_control"] == "True" else "GEO_non_control"
            by_path[key] += 1
        else:
            by_path["(no cohort used)"] += 1

        if r["files_saved"] == 0:
            no_files.append(r["disease"])

        t1 = r["trait_1"] if isinstance(r["trait_1"], int) else 0
        t0 = r["trait_0"] if isinstance(r["trait_0"], int) else 0
        # Saved means >=10
        if t0 >= 10 and t1 < 10:
            only_control.append((r["disease"], r["cohort_id"], t1, t0))
        elif t1 >= 10 and t0 < 10:
            only_disease.append((r["disease"], r["cohort_id"], t1, t0))

    print(f"\n{'='*70}")
    print(f"FILTER RUN SUMMARY  ({n} diseases)")
    print(f"{'='*70}\n")

    print("Status breakdown:")
    for k, v in sorted(by_status.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v:>4d}")

    print("\nPath breakdown:")
    for k, v in sorted(by_path.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v:>4d}")

    print(f"\nDiseases with NO csv files saved ({len(no_files)}):")
    for d in no_files:
        print(f"  {d}")

    print(f"\nDiseases with only CONTROL group (trait_1 < 10) ({len(only_control)}):")
    for d, c, t1, t0 in only_control:
        print(f"  {d:50s} cohort={c:18s} trait_1={t1:>4} trait_0={t0:>4}")

    print(f"\nDiseases with only DISEASE group (trait_0 < 10) ({len(only_disease)}):")
    for d, c, t1, t0 in only_disease:
        print(f"  {d:50s} cohort={c:18s} trait_1={t1:>4} trait_0={t0:>4}")


def main():
    rows = parse_log(LOG_PATH)
    write_csv(rows, CSV_OUT)
    print_summary(rows)
    print(f"\nFull per-disease table written to: {CSV_OUT}")


if __name__ == "__main__":
    main()

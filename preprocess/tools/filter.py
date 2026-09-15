import json
import logging
import os
import re
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _find_trait_col(df: pd.DataFrame, disease_name: str) -> Optional[str]:
    """Return the trait column if the first non-index column qualifies as one.
    A column qualifies when it's exactly the disease name (the TCGA convention,
    e.g. 'Bladder_Cancer') or carries an encoding annotation with parens and an
    equals sign (the GEO convention, e.g. 'Alopecia (control=0, disease=1)').
    Returns None when the dataset has no usable trait column - e.g. the header
    starts directly with 'Age'/'Gender' or with a gene symbol like 'A1BG'."""
    if len(df.columns) == 0:
        return None
    first = df.columns[0]
    if first in ('Age', 'Gender'):
        return None
    if first == disease_name:
        return first
    if '(' in first and '=' in first:
        return first
    return None


def _load_cohort_info(preprocess_dir: str, disease_name: str) -> dict:
    """Load cohort_info.json for a disease from the preprocess directory."""
    info_path = os.path.join(preprocess_dir, disease_name, "cohort_info.json")
    if os.path.isfile(info_path):
        with open(info_path, "r") as f:
            return json.load(f)
    return {}


def _coerce_bool(val) -> Optional[bool]:
    """Coerce a JSON value to bool. Accepts True/False, "TRUE"/"FALSE" (any case),
    1/0, "1"/"0". Returns None if the value cannot be interpreted."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and val in (0, 1):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "1"):
            return True
        if v in ("false", "0"):
            return False
    return None


def _get_trait_is_control(cohort_info: dict, cohort_name: str) -> bool:
    """Determine trait_is_control from cohort_info.json for a specific cohort.
    Falls back to checking mapping_note for 'control' keyword."""
    info = cohort_info.get(cohort_name, {})
    if not isinstance(info, dict):
        return False
    if "trait_is_control" in info:
        coerced = _coerce_bool(info["trait_is_control"])
        if coerced is not None:
            return coerced
    mn = info.get("trait_stats", {}).get("mapping_note", "")
    return "control" in mn.lower() if mn else False


def _format_val(val) -> str:
    """Format a trait value for use in a filename. Casts numeric floats like
    1.0 / 0.0 to clean ints; falls back to str() for everything else."""
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(val)


def _extract_trait_labels(trait_col: str) -> Dict[int, str]:
    """Extract value->label mapping from a trait column header. Preprocessing
    encodes the mapping inside parentheses, e.g. 'Alopecia (control=0, disease=1)'
    yields {0: 'control', 1: 'disease'}. Whitespace-tolerant, order-agnostic.
    Returns {} when the column has no parenthesized encoding (e.g. TCGA columns
    like 'Bladder_Cancer') or when parsing fails."""
    m = re.search(r"\(([^)]*)\)", trait_col)
    if not m:
        return {}
    result = {}
    for part in m.group(1).split(","):
        part = part.strip()
        if "=" not in part:
            continue
        label, _, val = part.rpartition("=")
        try:
            result[int(val.strip())] = label.strip()
        except ValueError:
            continue
    return result


def _sanitize_label(label: str) -> str:
    """Make a label safe for use in a filename. Keeps alphanumerics, underscore,
    hyphen, dot. Replaces everything else (slashes, spaces, parens, ...) with
    a single underscore."""
    safe = re.sub(r"[^\w\-.]+", "_", label).strip("_")
    return safe or "unknown"


def _save_subset(df, condition, clinical_cols, out_dir, filename, disease_name) -> int:
    """Save a subset (gene-only view) if it has >=10 samples. Returns sample count."""
    subset = df.loc[condition].copy()
    count = len(subset)
    print(f"{disease_name}: {filename} -> {count} samples")
    if count >= 10:
        os.makedirs(out_dir, exist_ok=True)
        gene_only = subset.drop(columns=clinical_cols, errors='ignore')
        gene_only.to_csv(os.path.join(out_dir, filename), index=True)
    return count


def _filter_one_dataset(csv_path: str, out_dir: str, cohort_id: str,
                        disease_name: str, trait_is_control: bool) -> Optional[int]:
    """Apply trait/Age/Gender filtering to one CSV.

    Returns:
      - None if the dataset is unusable (no detectable trait column).
      - Otherwise the primary group sample count: trait==1 count for the
        control case, or the largest group's count for the non-control case.
        The caller uses this to decide whether to fall back to the next cohort.
    """
    df = pd.read_csv(csv_path, index_col=0)

    trait_col = _find_trait_col(df, disease_name)
    if trait_col is None:
        head = list(df.columns[:3])
        print(f"{disease_name}: no trait column detected in {cohort_id} "
              f"(first columns: {head}); dataset unusable, skipped")
        return None

    age_col = "Age" if "Age" in df.columns else None
    gender_col = "Gender" if "Gender" in df.columns else None
    if age_col:
        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    if gender_col:
        df[gender_col] = pd.to_numeric(df[gender_col], errors="coerce")

    clinical_cols = [c for c in [trait_col, age_col, gender_col] if c is not None]

    if trait_is_control:
        primary_count = _save_subset(df, df[trait_col] == 1, clinical_cols, out_dir,
                                     f"{cohort_id}_{disease_name}_1.csv", disease_name)
        _save_subset(df, df[trait_col] == 0, clinical_cols, out_dir,
                     f"{cohort_id}_{disease_name}_0.csv", disease_name)
        base_cond = (df[trait_col] == 1)
    else:
        labels = _extract_trait_labels(trait_col)
        primary_count = 0
        for val in df[trait_col].dropna().unique():
            val_str = _format_val(val)
            try:
                label = labels.get(int(float(val)))
            except (TypeError, ValueError):
                label = None
            tag = f"{val_str}_{_sanitize_label(label)}" if label else val_str
            n = _save_subset(df, df[trait_col] == val, clinical_cols, out_dir,
                             f"{cohort_id}_{disease_name}_{tag}.csv",
                             disease_name)
            primary_count = max(primary_count, n)
        base_cond = pd.Series(True, index=df.index)

    # Age buckets: <40, [40, 60], >60 (60-year-olds go in the middle group)
    if age_col:
        _save_subset(df, base_cond & (df[age_col] < 40), clinical_cols, out_dir,
                     f"{cohort_id}_Age_under40.csv", disease_name)
        _save_subset(df, base_cond & (df[age_col] >= 40) & (df[age_col] <= 60),
                     clinical_cols, out_dir,
                     f"{cohort_id}_Age_40-60.csv", disease_name)
        _save_subset(df, base_cond & (df[age_col] > 60), clinical_cols, out_dir,
                     f"{cohort_id}_Age_over60.csv", disease_name)

    if gender_col:
        _save_subset(df, base_cond & (df[gender_col] == 1), clinical_cols, out_dir,
                     f"{cohort_id}_Gender_1.csv", disease_name)
        _save_subset(df, base_cond & (df[gender_col] == 0), clinical_cols, out_dir,
                     f"{cohort_id}_Gender_0.csv", disease_name)

    return primary_count


def _is_disease_completed(out_disease_dir: str) -> bool:
    return os.path.isfile(os.path.join(out_disease_dir, ".completed"))


def _mark_disease_completed(out_disease_dir: str):
    os.makedirs(out_disease_dir, exist_ok=True)
    with open(os.path.join(out_disease_dir, ".completed"), "w") as f:
        f.write("done")


def process_tcga_csv(csv_path, output_dir, disease_name):
    """Process a single TCGA.csv file. TCGA is always treated as control/disease,
    so trait_is_control is implicitly True regardless of cohort_info.json."""
    out_disease_dir = os.path.join(output_dir, disease_name)
    primary_count = _filter_one_dataset(csv_path, out_disease_dir, "TCGA",
                                        disease_name, trait_is_control=True)
    if primary_count is None:
        msg = (f"WARNING: {disease_name}: TCGA dataset has no usable trait column; "
               f"nothing was saved.")
        print(msg)
        logger.warning(msg)
    elif primary_count < 10:
        msg = (f"WARNING: {disease_name}: TCGA Disease=1 subset has only "
               f"{primary_count} samples (<10). Per the spec this should fall "
               f"back to GEO data, but auto-fallback is not implemented yet. "
               f"Inspect this disease manually if needed.")
        print(msg)
        logger.warning(msg)


def process_geo_csv(folder_A_path, folder_B_path, disease_name, trait_is_control=None):
    """Process GEO data for a disease. Tries cohorts in descending sample-size
    order; for the control case, falls back to the next cohort if the Disease=1
    subset has <10 samples (per spec 2.4). Also falls back when a cohort has no
    detectable trait column."""
    disease_path = os.path.join(folder_A_path, disease_name)

    tcga_file = os.path.join(disease_path, "TCGA.csv")
    if os.path.isfile(tcga_file):
        print(f"{disease_name}: skipped (TCGA exists)")
        return

    csv_files = [f for f in os.listdir(disease_path)
                 if f.lower().endswith(".csv") and f != "cohort_info.json"]
    if not csv_files:
        print(f"{disease_name}: no CSV files")
        return

    file_rows = {}
    for f in csv_files:
        path = os.path.join(disease_path, f)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                file_rows[f] = sum(1 for _ in fh) - 1
        except Exception:
            continue
    if not file_rows:
        return

    sorted_files = sorted(file_rows.items(), key=lambda x: x[1], reverse=True)
    cohort_info = _load_cohort_info(folder_A_path, disease_name)
    out_disease_dir = os.path.join(folder_B_path, disease_name)

    for selected_file, _ in sorted_files:
        cohort_id = os.path.splitext(selected_file)[0]
        csv_path = os.path.join(disease_path, selected_file)

        local_is_control = trait_is_control
        if local_is_control is None:
            local_is_control = _get_trait_is_control(cohort_info, cohort_id)

        print(f"{disease_name}: trying {selected_file} (trait_is_control={local_is_control})")
        primary_count = _filter_one_dataset(csv_path, out_disease_dir, cohort_id,
                                            disease_name, trait_is_control=local_is_control)

        if primary_count is None:
            print(f"{disease_name}: {selected_file} unusable (no trait), "
                  f"trying next-largest cohort")
            continue

        # Fallback rule (spec 2.4) only applies to the control case
        if local_is_control and primary_count < 10:
            print(f"{disease_name}: {selected_file} produced only {primary_count} "
                  f"Disease=1 samples (<10), falling back to next-largest cohort")
            continue
        print(f"{disease_name}: completed using {selected_file}")
        return

    msg = (f"WARNING: {disease_name}: exhausted all GEO cohorts; none produced "
           f"a usable Disease=1 subset with >=10 samples.")
    print(msg)
    logger.warning(msg)


def process_disease(in_data_root, output_root, trait):
    """Top-level entry. Auto-detects TCGA vs GEO for `trait` and dispatches to
    the appropriate filter, with an idempotent .completed checkpoint so reruns
    skip already-finished diseases."""
    out_disease_dir = os.path.join(output_root, trait)
    if _is_disease_completed(out_disease_dir):
        print(f"{trait}: already completed (checkpoint found), skipped")
        return

    disease_path = os.path.join(in_data_root, trait)
    if not os.path.isdir(disease_path):
        print(f"{trait}: input directory not found at {disease_path}, skipped")
        return

    tcga_file = os.path.join(disease_path, "TCGA.csv")
    if os.path.isfile(tcga_file):
        process_tcga_csv(tcga_file, output_root, trait)
    else:
        process_geo_csv(in_data_root, output_root, trait)

    _mark_disease_completed(out_disease_dir)

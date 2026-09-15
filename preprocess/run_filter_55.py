"""Standalone driver for the filter step. Walks every disease folder under
`in_data_root` and calls `tools.filter.process_disease` for each, writing
filtered subsets under `output_root`. Skips diseases whose `.completed`
checkpoint already exists, so reruns are cheap.

Usage:
    python run_filter_55.py
    python run_filter_55.py --in-root <preprocess_dir> --out-root <output_dir>
"""
import argparse
import os
import sys
import time
import traceback

from tools.filter import process_disease


class Tee:
    """Write to multiple streams at once so stdout shows up live AND lands in a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-root", default="./output/final-gpt41/preprocess",
        help="Directory containing preprocessed disease folders.",
    )
    parser.add_argument(
        "--out-root", default="./output/filter_55",
        help="Directory where filtered subsets will be written.",
    )
    args = parser.parse_args()

    in_root = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    log_path = os.path.join(out_root, "_run.log")
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    diseases = sorted(
        d for d in os.listdir(in_root)
        if os.path.isdir(os.path.join(in_root, d))
    )

    print(f"\n{'#' * 70}")
    print(f"Filter run started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  in_root  : {in_root}")
    print(f"  out_root : {out_root}")
    print(f"  diseases : {len(diseases)}")
    print(f"{'#' * 70}\n")

    n_done = n_skipped = n_error = 0
    errors = []
    t0 = time.time()

    for i, disease in enumerate(diseases, 1):
        marker = os.path.join(out_root, disease, ".completed")
        was_already_done = os.path.isfile(marker)
        print(f"\n[{i}/{len(diseases)}] {disease}")
        try:
            process_disease(in_root, out_root, disease)
            if was_already_done:
                n_skipped += 1
            else:
                n_done += 1
        except Exception as e:
            n_error += 1
            errors.append((disease, repr(e)))
            print(f"ERROR while processing {disease}: {e}")
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'#' * 70}")
    print(f"Filter run finished at {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"(elapsed: {elapsed/60:.1f} min)")
    print(f"  processed   : {n_done}")
    print(f"  skipped     : {n_skipped} (already had .completed marker)")
    print(f"  errors      : {n_error}")
    if errors:
        print(f"  error list  :")
        for d, msg in errors:
            print(f"    {d}: {msg}")
    print(f"  log         : {log_path}")
    print(f"{'#' * 70}\n")

    log_file.close()
    return 0 if n_error == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

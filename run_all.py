"""run_all.py -- one command to reproduce everything that needs no download.

The README used to hand out `/usr/bin/python3 ...` invocations, which is a
hard-coded interpreter path that exists on exactly one machine (not on
Windows, not in a venv, not in CI). This runs the no-download pipeline with
whatever interpreter is executing it, in the order the results depend on:

    unit tests -> synthetic dataset -> fit + evaluate -> demo.html
                                    -> real-pipeline smoke test

    python run_all.py            # everything
    python run_all.py --quick    # tests + smoke test only (~5s)

Anything needing a licensed download (RetailRocket, REES46, GA4) is listed at
the end rather than run -- see the README for those recipes.
"""

import argparse
import subprocess
import sys
import time

PY = sys.executable

FULL = [
    ("unit tests", [PY, "-m", "pytest", "-q"]),
    ("synthetic event dataset", [PY, "make_event_dataset.py"]),
    ("fit temperature + gate, evaluate", [PY, "evaluate_synthetic.py"]),
    ("demo.html", [PY, "demo.py"]),
    ("real-pipeline smoke test (no download)", [PY, "smoke_test_real_pipeline.py"]),
]

QUICK = [FULL[0], FULL[4]]

NEEDS_DATA = [
    ("RetailRocket (~900MB, Kaggle)",
     "python build_real_sessions.py --archive archive --out real_sessions.csv"
     " && python evaluate_real.py --data real_sessions.csv"),
    ("REES46 (~4.6GB, rees46.com)",
     "python build_rees46_sessions.py && python evaluate_rees46.py"),
    ("dwell / ablation / baselines / learned-headroom (need the above)",
     "python measure_dwell.py; python evaluate_ablation.py;"
     " python evaluate_baselines.py; python evaluate_learned.py"),
    ("GA4 cold-start prior (BigQuery export or the committed cell table)",
     "python ga4_prior_calibration.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tests + smoke test only")
    args = ap.parse_args()

    steps = QUICK if args.quick else FULL
    failures = []
    for i, (label, cmd) in enumerate(steps, 1):
        print(f"\n{'=' * 72}\n[{i}/{len(steps)}] {label}\n"
              f"    $ {' '.join(cmd)}\n{'=' * 72}", flush=True)
        t = time.time()
        rc = subprocess.call(cmd)
        status = "ok" if rc == 0 else f"FAILED (exit {rc})"
        print(f"-- {label}: {status} in {time.time() - t:.1f}s", flush=True)
        if rc != 0:
            failures.append(label)

    print(f"\n{'=' * 72}")
    if failures:
        print("FAILED steps: " + ", ".join(failures))
    else:
        print("all steps passed")
    print("\nNot run here (needs a licensed download):")
    for label, cmd in NEEDS_DATA:
        print(f"  - {label}\n      {cmd}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""
test.py - Demonstrates running α,β-Crown on a SimpleMLP MNIST model.

Model: SimpleMLP (784 -> 64 -> ReLU -> 10)
Dataset: MNIST
Verification property: L-inf robustness with epsilon=0.03

Usage:
    python test.py
"""

import subprocess
import sys
import os
import pickle
import time

# Path to abcrown.py and the config file
ABCROWN_DIR = os.path.join(os.path.dirname(__file__), "alpha-beta-CROWN", "complete_verifier")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "mnist_mlp_verify.yaml")
OUT_PATH = os.path.join(ABCROWN_DIR, "out.txt")


def run_verification():
    """Run α,β-Crown verifier using the YAML config file."""
    print("=" * 60)
    print("α,β-Crown Verification: SimpleMLP on MNIST")
    print("Model   : my_mlp.onnx (784 -> 64 -> ReLU -> 10)")
    print("Dataset : MNIST (50 test samples)")
    print("Property: L-inf robustness, epsilon=0.03")
    print("=" * 60)

    start = time.time()

    print("Running verifier... (this may take a moment)")

    # Run abcrown.py as a subprocess with the config file
    # stdout/stderr captured via PIPE — results are parsed from out.txt
    subprocess.run(
        [sys.executable, "-W", "ignore", "abcrown.py", "--config", CONFIG_PATH],
        cwd=ABCROWN_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    elapsed = time.time() - start
    print(f"\nTotal verification time: {elapsed:.2f}s")
    return elapsed


def parse_results():
    """Parse and display the verification results from out.txt."""
    if not os.path.exists(OUT_PATH):
        print("[ERROR] out.txt not found. Did the verifier run correctly?")
        sys.exit(1)

    with open(OUT_PATH, "rb") as f:
        data = pickle.load(f)

    summary = dict(data["summary"])
    results = data["results"]

    # Count each outcome
    safe_incomplete = summary.get("safe-incomplete", [])
    unsafe_pgd      = summary.get("unsafe-pgd", [])
    timeout         = summary.get("timeout", [])

    print("\n" + "=" * 60)
    print("Verification Results Summary")
    print("=" * 60)
    print(f"  Total samples     : {len(results)}")
    print(f"  safe-incomplete   : {len(safe_incomplete)}  (verified safe by α-Crown)")
    print(f"  unsafe-pgd        : {len(unsafe_pgd)}  (falsified by PGD attack)")
    print(f"  timeout           : {len(timeout)}  (not concluded within time limit)")

    # Per-sample breakdown
    print("\n" + "-" * 60)
    print(f"{'Sample':>8}  {'Result':<20}  {'Time (s)':>10}")
    print("-" * 60)
    for i, (status, t) in enumerate(results):
        print(f"{i:>8}  {status:<20}  {t:>10.4f}")

    # Average times per category
    safe_times   = [t for s, t in results if s == "safe-incomplete"]
    unsafe_times = [t for s, t in results if s == "unsafe-pgd"]

    print("\n" + "-" * 60)
    print("Average verification time:")
    if safe_times:
        print(f"  safe-incomplete : {sum(safe_times)/len(safe_times):.4f}s")
    if unsafe_times:
        print(f"  unsafe-pgd      : {sum(unsafe_times)/len(unsafe_times):.4f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
    parse_results()

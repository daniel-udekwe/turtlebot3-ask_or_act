#!/usr/bin/env python3
"""
Gazebo validation for the Ask-or-Act study (ANALYSIS_PLAN.md, section 7).

Writes the validation scenarios (final seeds 7000..7004 in each condition block:
5 per family, nominal and noisy = 80 scenarios) to scenarios/validation_labels.json
and runs them in Gazebo with four methods (320 episodes) through run_episodes.py.
Results go to runs/validation/results.csv. Resumable: rerun to continue.

Afterwards compare with the offline simulator:
  python3 compare_validation.py

Usage:
  python3 gazebo_validation.py
  python3 gazebo_validation.py --methods always_ask          # a subset
  python3 gazebo_validation.py --only cafe_hs_x_7000 --out ~/ask_or_act/runs/validation_test   # quick test
"""
import argparse
import json
import os
import subprocess
import sys

from generate_scenarios import ENVS, FAMILIES, build, eval_seeds

HOME = os.path.expanduser('~')
BASE = os.path.join(HOME, 'ask_or_act')
HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(BASE, 'scenarios', 'validation_labels.json')
OUT = os.path.join(BASE, 'runs', 'validation')
CONDITIONS = ['nominal', 'noisy']
SEED_START, N_SEEDS = 7000, 5

# method name -> run_episodes.py arguments
METHODS = {
    'always_act': ['--policy', 'always_act'],
    'always_ask': ['--policy', 'always_ask'],
    'threshold:raw': ['--policy', 'threshold', '--calibration', 'raw'],
    'threshold:isotonic_dw@recal': ['--policy', 'threshold', '--calibration', 'isotonic_dw',
                                    '--calib-condition', 'recal'],
}


def write_labels():
    labels = [build(e, f, i, s, c) for c in CONDITIONS for e in ENVS for f, i in FAMILIES
              for s in eval_seeds(c, SEED_START, N_SEEDS)]
    os.makedirs(os.path.dirname(LABELS), exist_ok=True)
    with open(LABELS, 'w') as f:
        json.dump(labels, f, indent=2)
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
    ap.add_argument('--only', default='*', help="scenario id pattern, e.g. 'cafe_hs_x_7000'")
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    labels = write_labels()
    print(f'{len(labels)} validation scenarios written to {LABELS}')
    for n, m in enumerate(args.methods, 1):
        print(f'\n=== [{n}/{len(args.methods)}] method {m} ===')
        subprocess.run([sys.executable, os.path.join(HERE, 'run_episodes.py'), '--labels', LABELS,
                        '--out', args.out, '--method', m, '--resume', '--only', args.only] + METHODS[m], check=True)
    print(f'\nDone. Results in {args.out}/results.csv. Next: python3 compare_validation.py')


if __name__ == '__main__':
    main()

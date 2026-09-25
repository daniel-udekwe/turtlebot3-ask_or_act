#!/usr/bin/env python3
"""
Offline dataset builder for the Ask-or-Act intent predictor.

Generates scenarios with the same generator as the Gazebo experiments, but with
separate seed ranges, computes each pedestrian's trajectory with ped_traj.py
(no Gazebo needed), simulates noisy tracking at OBS_RATE, and writes one row
per decision time with the features and the ground-truth intent.

Splits (seed ranges never overlap each other or the Gazebo seeds 0..99):
  train          nominal condition only, seeds 1000..
  calib@<cond>   per condition, for fitting (re)calibration
  test@<cond>    per condition, for final evaluation
The predictor is always trained on nominal data; the other conditions are
distribution shifts (see CONDITIONS in generate_scenarios.py).

Decision window: from release until the moment a crossing pedestrian reaches the
robot's line (the same window for both twins of an ambiguous pair), since after
that the intent is no longer uncertain.

Usage:
  python3 build_dataset.py                     # all conditions
  python3 build_dataset.py --conditions nominal slow
  python3 build_dataset.py --train-seeds 100 --calib-seeds 10
"""
import argparse
import csv
import os
import random

from generate_scenarios import CONDITIONS, ENVS, FAMILIES, build
from intent_features import FEATURE_NAMES, compute_features
from ped_traj import build_trajectory, pose_at

HOME = os.path.expanduser('~')
OUT = os.path.join(HOME, 'ask_or_act', 'predictor')

OBS_RATE = 10.0          # Hz, tracker update rate
PRE_RELEASE = 2.0        # s of standing-still history before release
NONCROSS_WINDOW = 5.0    # s, decision window for clear non-crossing scenarios


def episode_samples(label, rng, noise):
    traj = build_trajectory(label['start'], label['plan'], label['human_speed'])
    times = [s[0] for s in traj]
    if label['commit_t'] is not None:
        t_end = label['commit_t']
    elif label.get('window_end') is not None:
        t_end = label['window_end']
    else:
        t_end = NONCROSS_WINDOW
    dt = 1.0 / OBS_RATE
    hist, rows = [], []
    k = -int(round(PRE_RELEASE * OBS_RATE))
    while True:
        t = k * dt
        if t > t_end + 1e-9:
            break
        x, y, _, _ = pose_at(traj, times, t)
        hist.append((t, x + rng.gauss(0.0, noise), y + rng.gauss(0.0, noise)))
        if t >= 0.0:
            f = compute_features(hist)
            if f is not None:
                rows.append(dict(t=round(t, 2), time_to_commit=round(t_end - t, 2),
                                 **{n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, f)}))
        k += 1
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train-seeds', type=int, default=50)
    ap.add_argument('--calib-seeds', type=int, default=10,
                    help='seeds per condition for calibration (a small set, as a robot would collect)')
    ap.add_argument('--test-seeds', type=int, default=20, help='seeds per condition for testing')
    ap.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument('--test-seed-start', type=int, default=3000, help='start of each condition\'s test seed block')
    ap.add_argument('--only-test', action='store_true', help='write only the test splits (for a fresh final test set)')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    conds = ['nominal'] + [c for c in args.conditions if c != 'nominal']
    splits = {} if args.only_test else {'train': ('nominal', range(1000, 1000 + args.train_seeds))}
    for k, c in enumerate(CONDITIONS):                 # fixed index keeps seed ranges stable
        if c in conds:
            base = 10000 * k
            if not args.only_test:
                splits[f'calib@{c}'] = (c, range(base + 2000, base + 2000 + args.calib_seeds))
            ts = args.test_seed_start
            splits[f'test@{c}'] = (c, range(base + ts, base + ts + args.test_seeds))
    path = os.path.join(args.out, 'dataset.csv')
    cols = ['split', 'condition', 'scenario', 'env', 'family', 'intent', 'label', 'ambiguous',
            't', 'time_to_commit'] + FEATURE_NAMES
    counts = {}
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for split, (cond, seeds) in splits.items():
            n = 0
            for env in ENVS:
                for family, intent in FAMILIES:
                    for seed in seeds:
                        lab = build(env, family, intent, seed, cond)
                        rng = random.Random(f"noise-{lab['id']}")
                        for r in episode_samples(lab, rng, CONDITIONS[cond]['noise']):
                            w.writerow(dict(split=split, condition=cond, scenario=lab['id'], env=env,
                                            family=family, intent=intent, label=int(intent == 'crossing'),
                                            ambiguous=lab['ambiguous'], **r))
                            n += 1
            counts[split] = n
    print(f'Wrote {path}')
    for split, n in counts.items():
        print(f'  {split:<20} {n:6d} samples from {len(splits[split][1]) * len(FAMILIES) * len(ENVS)} scenarios')


if __name__ == '__main__':
    main()

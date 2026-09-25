#!/usr/bin/env python3
"""
Offline dataset builder for the Ask-or-Act intent predictor.

Generates scenarios with the same generator as the Gazebo experiments, but with
separate seed ranges, computes each pedestrian's trajectory with ped_traj.py
(no Gazebo needed), simulates noisy tracking at OBS_RATE, and writes one row
per decision time with the features and the ground-truth intent.

Splits (seed ranges never overlap each other or the Gazebo seeds 0..99):
  train  seeds 1000..1000+N-1
  calib  seeds 2000..2000+M-1   (fitting temperature scaling / isotonic calibration)
  test   seeds 3000..3000+M-1   (final evaluation)

Decision window: from release until the moment a crossing pedestrian reaches the
robot's line (the same window for both twins of an ambiguous pair), since after
that the intent is no longer uncertain.

Usage:
  python3 build_dataset.py                    # 50 train seeds, 20 calib, 20 test
  python3 build_dataset.py --train-seeds 100
"""
import argparse
import csv
import os
import random

from generate_scenarios import ENVS, FAMILIES, build
from intent_features import FEATURE_NAMES, compute_features
from ped_traj import build_trajectory, pose_at

HOME = os.path.expanduser('~')
OUT = os.path.join(HOME, 'ask_or_act', 'predictor')

OBS_RATE = 10.0          # Hz, tracker update rate
POS_NOISE = 0.03         # m, standard deviation of tracked position noise
PRE_RELEASE = 2.0        # s of standing-still history before release
NONCROSS_WINDOW = 5.0    # s, decision window for clear non-crossing scenarios


def episode_samples(label, rng):
    traj = build_trajectory(label['start'], label['plan'], label['human_speed'])
    times = [s[0] for s in traj]
    t_end = label['commit_t'] if label['commit_t'] is not None else NONCROSS_WINDOW
    dt = 1.0 / OBS_RATE
    hist, rows = [], []
    k = -int(round(PRE_RELEASE * OBS_RATE))
    while True:
        t = k * dt
        if t > t_end + 1e-9:
            break
        x, y, _, _ = pose_at(traj, times, t)
        hist.append((t, x + rng.gauss(0.0, POS_NOISE), y + rng.gauss(0.0, POS_NOISE)))
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
    ap.add_argument('--eval-seeds', type=int, default=20, help='seeds for each of calib and test')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    splits = dict(train=range(1000, 1000 + args.train_seeds),
                  calib=range(2000, 2000 + args.eval_seeds),
                  test=range(3000, 3000 + args.eval_seeds))
    path = os.path.join(args.out, 'dataset.csv')
    cols = ['split', 'scenario', 'env', 'family', 'intent', 'label', 'ambiguous', 't', 'time_to_commit'] + FEATURE_NAMES
    counts = {}
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for split, seeds in splits.items():
            n = 0
            for env in ENVS:
                for family, intent in FAMILIES:
                    for seed in seeds:
                        lab = build(env, family, intent, seed)
                        rng = random.Random(f"noise-{lab['id']}")
                        for r in episode_samples(lab, rng):
                            w.writerow(dict(split=split, scenario=lab['id'], env=env, family=family,
                                            intent=intent, label=int(intent == 'crossing'),
                                            ambiguous=lab['ambiguous'], **r))
                            n += 1
            counts[split] = n
    print(f'Wrote {path}')
    for split, n in counts.items():
        print(f'  {split:<6} {n:6d} samples from {len(splits[split]) * len(FAMILIES) * len(ENVS)} scenarios')


if __name__ == '__main__':
    main()

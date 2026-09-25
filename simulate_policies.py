#!/usr/bin/env python3
"""
Offline closed-loop simulator for the Ask-or-Act study.

Replicates a Gazebo episode without Gazebo: the robot drives the line at constant
speed (or stops), the pedestrian follows its deterministic trajectory, the robot
tracks it at 10 Hz with the condition's noise, and asks are answered by the same
seeded response model as scripted_pedestrians.py. The decision logic is the same
policy.py used by line_controller.py, so results are directly comparable; Gazebo
runs validate a subset.

Each method is a policy, optionally with a predictor calibration:
  always_act, always_cautious, always_ask, oracle
  threshold:raw          raw predictor
  threshold:<cal>@stale  calibrator fitted on nominal data (cal = isotonic or temperature)
  threshold:<cal>@recal  calibrator fitted on the test condition's own calibration set

Usage:
  python3 simulate_policies.py
  python3 simulate_policies.py --conditions nominal noisy --seeds 50
  python3 simulate_policies.py --p-low 0.1 --p-high 0.9 --response-delay 3.0
  python3 simulate_policies.py --summary-only        # re-print the summary of the last run
"""
import argparse
import math
import os
import random

import pandas as pd

from generate_scenarios import CONDITIONS, ENVS, FAMILIES, ROBOT, build
from ped_traj import HUMAN_RADIUS, build_trajectory, pose_at
from policy import AskOrActPolicy, load_predictor

HOME = os.path.expanduser('~')
BASE = os.path.join(HOME, 'ask_or_act')
DT = 0.05               # controller step (20 Hz)
OBS_DT = 0.1            # tracker rate (10 Hz)
START_DELAY = 2.0       # controller observes this long before the episode starts
MAX_DURATION = 60.0
ROBOT_RADIUS = 0.1          # matches COLLISION_DIST = 0.4 in run_episodes.py
COLLISION_DIST = HUMAN_RADIUS + ROBOT_RADIUS
DEFAULT_METHODS = ['always_act', 'always_cautious', 'always_ask', 'oracle',
                   'threshold:raw', 'threshold:isotonic@stale', 'threshold:isotonic@recal']


def simulate(label, pol, noise, response_delay, response_prob, answer_accuracy):
    traj = build_trajectory(label['start'], label['plan'], label['human_speed'])
    times = [s[0] for s in traj]
    rng_obs = random.Random(f"obs-{label['id']}")
    rng_ans = random.Random(f"answers-{label['id']}")
    (x0, y0), (xg, yg) = ROBOT['start'], ROBOT['goal']
    rx, speed = x0, ROBOT['speed']
    release, rdelay = label['release_delay'], label['robot_delay']
    pending, stopped, min_d, cross, prev_hy = [], 0.0, float('inf'), None, None
    n_steps_obs = int(round(OBS_DT / DT))
    k0 = -int(round(START_DELAY / DT))
    outcome, t = 'timed out', 0.0
    for k in range(k0, int(MAX_DURATION / DT) + 1):
        t = k * DT
        hx, hy, _, _ = pose_at(traj, times, t - release)
        if k % n_steps_obs == 0:
            pol.add_observation(1, t, hx + rng_obs.gauss(0, noise), hy + rng_obs.gauss(0, noise))
        if t < 0:
            continue
        for item in [a for a in pending if a[0] <= t]:
            pending.remove(item)
            pol.receive_answer(item[1], item[2], t)
        ry = y0 + (yg - y0) * (x0 - rx) / (x0 - xg)
        min_d = min(min_d, math.hypot(hx - rx, hy - ry))
        if prev_hy is not None and t >= release and (prev_hy < 0) != (hy < 0) and cross is None:
            cross = rx - hx
        prev_hy = hy
        if rx - xg <= 0.15:
            outcome = 'reached goal'
            break
        if t < rdelay:
            continue
        go, _, to_ask = pol.decide(t, rx)
        for hid in to_ask:
            if rng_ans.random() < response_prob:
                wrong = 'not_crossing' if label['intent'] == 'crossing' else 'crossing'
                ans = label['intent'] if rng_ans.random() < answer_accuracy else wrong
                pending.append((t + response_delay, hid, ans))
        if go:
            rx -= speed * DT
        else:
            stopped += DT
    free_time = rdelay + (x0 - xg - 0.15) / speed
    return dict(outcome=outcome, duration=round(t, 3), delay=round(t - free_time, 3), min_dist=round(min_d, 3),
                collision=min_d < COLLISION_DIST, n_asks=len(pol.asks), stopped_time=round(stopped, 3),
                robot_dist_at_cross=None if cross is None else round(cross, 3))


def make_policy(method, label, predictors, args, condition):
    name, _, variant = method.partition(':')
    common = dict(p_low=args.p_low, p_high=args.p_high, engage_dist=args.engage_dist, ask_dist=args.ask_dist,
                  true_intent=label['intent'])
    if name != 'threshold':
        return AskOrActPolicy(name, **common)
    if variant == 'raw':
        model, cal = predictors[('raw', None)]
    else:
        cal_kind, _, which = variant.partition('@')
        model, cal = predictors[(cal_kind, 'nominal' if which == 'stale' else condition)]
    return AskOrActPolicy('threshold', model, cal, **common)


def summarize(df, args):
    """Print and save a summary per condition and method (works with old and new pandas)."""
    df = df.copy()
    for col in ('collision', 'ambiguous'):
        df[col] = df[col].astype(str).str.lower().eq('true')
    df['asked'] = df['n_asks'] > 0
    out = []
    for (cond, method), g in df.groupby(['condition', 'method'], sort=False):
        out.append(dict(condition=cond, method=method,
                        collision_rate=g['collision'].mean(),
                        collisions_on_crossing=g.loc[g.intent == 'crossing', 'collision'].mean(),
                        asks_per_episode=g['n_asks'].mean(),
                        ask_rate_clear=g.loc[~g.ambiguous, 'asked'].mean(),
                        ask_rate_ambiguous=g.loc[g.ambiguous, 'asked'].mean(),
                        mean_delay_s=g['delay'].mean(),
                        stop_s_not_crossing=g.loc[g.intent == 'not_crossing', 'stopped_time'].mean()))
    summ = pd.DataFrame(out).set_index(['condition', 'method'])
    pd.set_option('display.width', 180)
    print(f'\nSummary (band {args.p_low}-{args.p_high}, deadline {args.ask_dist} m, response delay '
          f'{args.response_delay} s, response prob {args.response_prob}, answer accuracy {args.answer_accuracy})')
    print(summ.round(3).to_string())
    summ.to_csv(args.out.replace('.csv', '_summary.csv'))
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    ap.add_argument('--conditions', nargs='+', default=['nominal', 'noisy', 'combined'], choices=list(CONDITIONS))
    ap.add_argument('--seeds', type=int, default=30, help='scenarios per family/intent/condition')
    ap.add_argument('--seed-start', type=int, default=5000, help='offset added to 10000*condition index')
    ap.add_argument('--model', default='mlp')
    ap.add_argument('--model-file', default=os.path.join(BASE, 'predictor', 'intent_model.pkl'))
    ap.add_argument('--p-low', type=float, default=0.2)
    ap.add_argument('--p-high', type=float, default=0.8)
    ap.add_argument('--engage-dist', type=float, default=1.0)
    ap.add_argument('--ask-dist', type=float, default=0.6, help='decision deadline (m ahead)')
    ap.add_argument('--response-delay', type=float, default=1.5)
    ap.add_argument('--response-prob', type=float, default=1.0)
    ap.add_argument('--answer-accuracy', type=float, default=1.0)
    ap.add_argument('--out', default=os.path.join(BASE, 'sim', 'results.csv'))
    ap.add_argument('--summary-only', action='store_true', help='summarize an existing --out file and exit')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.summary_only:
        summarize(pd.read_csv(args.out), args)
        return

    rows = []
    cond_index = {c: i for i, c in enumerate(CONDITIONS)}
    for cond in args.conditions:
        predictors = {('raw', None): load_predictor(args.model_file, args.model, 'raw')}
        for kind in ('isotonic', 'temperature'):
            for c in {'nominal', cond}:
                predictors[(kind, c)] = load_predictor(args.model_file, args.model, kind, c)
        seeds = range(10000 * cond_index[cond] + args.seed_start, 10000 * cond_index[cond] + args.seed_start + args.seeds)
        for env in ENVS:
            for family, intent in FAMILIES:
                for seed in seeds:
                    label = build(env, family, intent, seed, cond)
                    for method in args.methods:
                        pol = make_policy(method, label, predictors, args, cond)
                        r = simulate(label, pol, CONDITIONS[cond]['noise'], args.response_delay,
                                     args.response_prob, args.answer_accuracy)
                        rows.append(dict(id=label['id'], condition=cond, method=method, family=family, intent=intent,
                                         ambiguous=label['ambiguous'], **r))
        print(f'{cond}: {len(seeds) * len(FAMILIES) * len(ENVS)} scenarios x {len(args.methods)} methods done')

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summarize(df, args)
    print(f'\nPer-episode results in {args.out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Scenario generator for the Ask-or-Act study (scripted pedestrians).

Writes ~/ask_or_act/scenarios/labels.json and manifest.csv. Each scenario has
one pedestrian whose motion is fully defined by a plan (see ped_traj.py), its
ground-truth intent, and the start offsets that make the encounter happen at
the intended moment.

Scenario families (robot drives a fixed line along y ~ 0):
  clear_cross     walks straight across the robot's line
  clear_noncross  walks away, parallel to the line, never enters it
  hesitant        walks toward the line, pauses, then crosses or turns back      (ambiguous)
  late_turn       heads for the line, then crosses or veers away at the last moment (ambiguous)
  slow_cross      crosses slowly and steadily; readable before the robot must decide
  wait_far        walks toward the line, stops well back and waits; readable before the robot must decide

Timing: for ambiguous scenarios, crossing and not-crossing twins with the same
seed share every parameter and the same start offsets, computed from the
crossing version: the robot is `robot_dist_at_commit` m from the crossing point
when a crossing pedestrian would reach the line. So timing alone never reveals
the intent.

Usage:
  python3 generate_scenarios.py --seeds 5
  python3 generate_scenarios.py --seeds 5 --conditions nominal slow combined
  python3 generate_scenarios.py --seeds 5 --conditions nominal noisy --seed-start 7000
"""
import argparse
import csv
import json
import os
import random

from ped_traj import build_trajectory, first_line_crossing

HOME = os.path.expanduser('~')
DEFAULT_LABELS = os.path.join(HOME, 'ask_or_act', 'scenarios')

# Must match line_controller.py and the robot spawn
ROBOT = dict(start=(2.9, 0.1), goal=(-2.9, 0.0), speed=0.18)

# Walkable reference points per environment (checked against the map with plot_map.py)
ENVS = {
    'cafe': dict(
        south=(1.5, -2.6),       # pedestrian start, south of the robot's line
        north=(1.5, 3.5),        # far side of the line (crossing target)
        west=(-3.5, -3.2),       # walks away, parallel to the line (clear non-crossing)
        veer=(3.3, -2.5),        # late-turn escape, away from the line
    ),
}

# Experimental conditions. 'nominal' is the training distribution; the others are
# distribution shifts used to test calibration. `noise` is the tracking noise (m)
# used by build_dataset.py; the rest change how pedestrians behave.
NOMINAL = dict(speed=(0.8, 1.2), pause=(1.5, 2.5), hes_gap=(0.8, 1.2), late_gap=(0.4, 0.7),
               slow_speed=(0.35, 0.55), wait_gap=(1.6, 2.2), noise=0.03)
CONDITIONS = {
    'nominal': NOMINAL,
    'slow': dict(NOMINAL, speed=(0.5, 0.8)),                 # slower walkers
    'fast': dict(NOMINAL, speed=(1.2, 1.6)),                 # faster walkers
    'long_pause': dict(NOMINAL, pause=(3.0, 5.0)),           # longer hesitation
    'noisy': dict(NOMINAL, noise=0.10),                      # poorer tracking
    'combined': dict(NOMINAL, speed=(0.5, 0.8), pause=(3.0, 5.0), noise=0.10),
}

def eval_seeds(condition, start, n):
    """Seed range for a condition: each condition has its own block of 10000 seeds,
    so different conditions never reuse the same random draws."""
    base = 10000 * list(CONDITIONS).index(condition)
    return range(base + start, base + start + n)


FAMILIES = [
    ('clear_cross', 'crossing'),
    ('clear_noncross', 'not_crossing'),
    ('hesitant', 'crossing'),
    ('hesitant', 'not_crossing'),
    ('late_turn', 'crossing'),
    ('late_turn', 'not_crossing'),
    ('slow_cross', 'crossing'),
    ('wait_far', 'not_crossing'),
]
SHORT = {'clear_cross': 'cc', 'clear_noncross': 'cn', 'hesitant': 'hs', 'late_turn': 'lt',
         'slow_cross': 'sc', 'wait_far': 'wf', 'crossing': 'x', 'not_crossing': 'n'}
AMBIGUOUS = ('hesitant', 'late_turn')
WAIT_AHEAD = (1.2, 1.8)     # wait_far: robot this far (m) from the pedestrian when it stops


def walk(p, stop=True):
    return dict(type='walk', to=[round(p[0], 3), round(p[1], 3)], stop=stop)


def point_at_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1])
    return (a[0] + t * (b[0] - a[0]), y)


def plans(env, family, rng, cond):
    """Return (start, params, plan_for(intent)). Random draws are shared by both intents."""
    dx = rng.uniform(-0.2, 0.2)                    # small shift of the route, for variety
    S = (env['south'][0] + dx, env['south'][1])
    N = (env['north'][0] + dx, env['north'][1])
    params = dict(route_shift=round(dx, 3))
    if family == 'clear_cross':
        return S, params, lambda intent: [walk(N)]
    if family == 'clear_noncross':
        return S, params, lambda intent: [walk(env['west'])]
    if family == 'hesitant':
        gap = rng.uniform(*cond['hes_gap'])        # stops this far before the line (m)
        pause = rng.uniform(*cond['pause'])        # pause duration (s)
        A = point_at_y(S, N, -gap)
        params.update(gap=round(gap, 3), pause=round(pause, 3))
        return S, params, lambda intent: [walk(A), dict(type='wait', duration=round(pause, 3)),
                                          walk(N if intent == 'crossing' else S)]
    if family == 'late_turn':
        gap = rng.uniform(*cond['late_gap'])       # turns this far before the line (m)
        B = point_at_y(S, N, -gap)
        params.update(gap=round(gap, 3))
        return S, params, lambda intent: [walk(B, stop=False),
                                          walk(N if intent == 'crossing' else env['veer'])]
    if family == 'slow_cross':
        return S, params, lambda intent: [walk(N)]
    if family == 'wait_far':
        gap = rng.uniform(*cond['wait_gap'])       # stops this far before the line (m)
        F = point_at_y(S, N, -gap)
        params.update(gap=round(gap, 3))
        return S, params, lambda intent: [walk(F), dict(type='wait', duration=120.0)]
    raise ValueError(family)


def offsets(commit_t, commit_x, d_commit):
    """Start offsets so the robot is d_commit m short of commit_x when the pedestrian commits."""
    target_x = commit_x + d_commit
    rel = commit_t - (ROBOT['start'][0] - target_x) / ROBOT['speed']   # robot start - release
    return (0.0, rel) if rel >= 0.0 else (-rel, 0.0)                   # (release_delay, robot_delay)


def build(env_name, family, intent, seed, condition='nominal'):
    env = ENVS[env_name]
    cond = CONDITIONS[condition]
    rng = random.Random(f'{env_name}-{family}-{seed}')     # same draws for both intents
    speed = rng.uniform(*cond['speed'])                     # walking speed (m/s)
    d_commit = rng.uniform(0.0, 0.4)                        # robot distance to crossing point at commit
    start, params, plan_for = plans(env, family, rng, cond)
    plan = plan_for(intent)
    window_end = None

    if family == 'slow_cross':
        speed = rng.uniform(*cond['slow_speed'])
    if family == 'clear_noncross':
        commit = None
        release, robot_delay = rng.uniform(0.0, 4.0), 0.0
    elif family == 'wait_far':
        # Release so the pedestrian stops when the robot is WAIT_AHEAD m away from it
        commit = None
        traj = build_trajectory(start, plan, speed)
        t_stop = next(tt for tt, _, _, _, v in reversed(traj) if v > 0.0)
        stop_x = plan[0]['to'][0]
        ahead = rng.uniform(*WAIT_AHEAD)
        rel = t_stop - (ROBOT['start'][0] - (stop_x + ahead)) / ROBOT['speed']
        release, robot_delay = (0.0, rel) if rel >= 0.0 else (-rel, 0.0)
        window_end = t_stop + (ahead + 0.5) / ROBOT['speed']     # until the robot has passed
    else:
        commit = first_line_crossing(build_trajectory(start, plan_for('crossing'), speed))
        release, robot_delay = offsets(commit[0], commit[1], d_commit)

    sid = f"{env_name}_{SHORT[family]}_{SHORT[intent]}_{seed:02d}"
    if condition != 'nominal':
        sid += f'_{condition}'
    return dict(
        id=sid, env=env_name, family=family, intent=intent, condition=condition,
        ambiguous=family in AMBIGUOUS, seed=seed,
        human_speed=round(speed, 3), robot_dist_at_commit=round(d_commit, 3),
        release_delay=round(release, 3), robot_delay=round(robot_delay, 3),
        commit_t=round(commit[0], 3) if commit else None,
        commit_x=round(commit[1], 3) if commit else None,
        window_end=round(window_end, 3) if window_end is not None else None,
        start=[round(start[0], 3), round(start[1], 3)], plan=plan,
        waypoints=[s['to'] for s in plan if s['type'] == 'walk'], **params)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=5, help='scenarios per family/intent/env')
    ap.add_argument('--envs', nargs='+', default=['cafe'], choices=sorted(ENVS))
    ap.add_argument('--conditions', nargs='+', default=['nominal'], choices=list(CONDITIONS))
    ap.add_argument('--seed-start', type=int, default=None,
                    help='use seeds from eval_seeds(condition, start, n), e.g. 7000 for the final evaluation; '
                         'default: seeds 0..n-1 (the original Gazebo scenarios)')
    ap.add_argument('--labels-dir', default=DEFAULT_LABELS)
    args = ap.parse_args()
    os.makedirs(args.labels_dir, exist_ok=True)

    seeds = (lambda c: range(args.seeds)) if args.seed_start is None else \
            (lambda c: eval_seeds(c, args.seed_start, args.seeds))
    labels = [build(e, f, i, s, c) for c in args.conditions for e in args.envs
              for f, i in FAMILIES for s in seeds(c)]
    with open(os.path.join(args.labels_dir, 'labels.json'), 'w') as f:
        json.dump(labels, f, indent=2)
    cols = ['id', 'condition', 'env', 'family', 'intent', 'ambiguous', 'seed', 'human_speed',
            'robot_dist_at_commit', 'release_delay', 'robot_delay', 'commit_t', 'commit_x']
    with open(os.path.join(args.labels_dir, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(labels)
    print(f'Generated {len(labels)} scenarios; labels in {args.labels_dir}')


if __name__ == '__main__':
    main()

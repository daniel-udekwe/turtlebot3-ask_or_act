#!/usr/bin/env python3
"""
Compare the Gazebo validation runs with the offline simulator (ANALYSIS_PLAN.md, section 7).

Runs the offline simulator on exactly the same scenarios and methods as
gazebo_validation.py and matches the episodes one to one. Reports, per method and
condition and overall:
  collision agreement   % of episodes with the same collision outcome (criterion: >= 90%)
  collision rates       Gazebo vs offline
  minimum distance      Pearson correlation and mean absolute difference
  question counts       % of episodes with the same number of questions
and lists every episode whose collision outcome differs.

Outputs (in runs/validation/): comparison.csv, VALIDATION.md, fig_min_dist.png

Usage:
  python3 compare_validation.py
"""
import argparse
import json
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import simulate_policies as sp
from generate_scenarios import CONDITIONS
from policy import load_predictor

HOME = os.path.expanduser('~')
BASE = os.path.join(HOME, 'ask_or_act')
LABELS = os.path.join(BASE, 'scenarios', 'validation_labels.json')
RUNS = os.path.join(BASE, 'runs', 'validation')
CRITERION = 0.90


def offline(labels, methods, model_file):
    pargs = SimpleNamespace(p_low=0.2, p_high=0.8, engage_dist=1.0, ask_dist=0.6, smooth_n=5, yield_timeout=8.0)
    cache, rows = {}, []
    for lab in labels:
        cond = lab['condition']
        if cond not in cache:
            preds = {('raw', None): load_predictor(model_file, 'mlp', 'raw')}
            for kind in ('isotonic', 'isotonic_dw', 'temperature'):
                for c in {'nominal', cond}:
                    preds[(kind, c)] = load_predictor(model_file, 'mlp', kind, c)
            cache[cond] = preds
        for m in methods:
            pol = sp.make_policy(m, lab, cache[cond], pargs, cond)
            r = sp.simulate(lab, pol, CONDITIONS[cond]['noise'], 1.5, 1.0, 1.0)
            rows.append(dict(id=lab['id'], method=m, collision=r['collision'], min_dist=r['min_dist'],
                             n_asks=r['n_asks'], duration=r['duration']))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=RUNS)
    ap.add_argument('--labels', default=LABELS)
    ap.add_argument('--model-file', default=os.path.join(BASE, 'predictor', 'intent_model.pkl'))
    args = ap.parse_args()

    gz = pd.read_csv(os.path.join(args.runs, 'results.csv'))
    gz['collision'] = gz['collision'].astype(str).str.lower().eq('true')
    labels = [l for l in json.load(open(args.labels)) if l['id'] in set(gz['id'])]
    methods = list(dict.fromkeys(gz['method']))
    print(f'{len(gz)} Gazebo episodes, {len(labels)} scenarios, methods: {methods}')
    off = offline(labels, methods, args.model_file)

    m = gz[['id', 'method', 'condition', 'family', 'intent', 'collision', 'min_dist', 'n_asks', 'duration']].merge(
        off, on=['id', 'method'], suffixes=('_gazebo', '_offline'))
    m['same_collision'] = m['collision_gazebo'] == m['collision_offline']
    m['same_asks'] = m['n_asks_gazebo'] == m['n_asks_offline']
    m.to_csv(os.path.join(args.runs, 'comparison.csv'), index=False)

    def summary(g):
        r = np.corrcoef(g.min_dist_gazebo, g.min_dist_offline)[0, 1] if len(g) > 2 else float('nan')
        return dict(episodes=len(g), collision_agreement=g.same_collision.mean(),
                    collisions_gazebo=g.collision_gazebo.mean(), collisions_offline=g.collision_offline.mean(),
                    min_dist_r=r, min_dist_mad=(g.min_dist_gazebo - g.min_dist_offline).abs().mean(),
                    same_asks=g.same_asks.mean())

    rows = [dict(method=k[0], condition=k[1], **summary(g)) for k, g in m.groupby(['method', 'condition'], sort=False)]
    per = pd.DataFrame(rows)
    overall = summary(m)
    passed = overall['collision_agreement'] >= CRITERION
    dis = m[~m.same_collision][['id', 'method', 'intent', 'min_dist_gazebo', 'min_dist_offline']]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for meth in methods:
        g = m[m.method == meth]
        ax.scatter(g.min_dist_offline, g.min_dist_gazebo, s=14, label=meth, alpha=0.7)
    lim = [0, max(m.min_dist_offline.max(), m.min_dist_gazebo.max()) * 1.05]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.axhline(0.4, color='r', lw=0.8, ls=':')
    ax.axvline(0.4, color='r', lw=0.8, ls=':')
    ax.set_xlabel('minimum distance, offline simulator (m)')
    ax.set_ylabel('minimum distance, Gazebo (m)')
    ax.set_title(f"Gazebo vs offline (r = {overall['min_dist_r']:.3f})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.runs, 'fig_min_dist.png'), dpi=130)

    with open(os.path.join(args.runs, 'VALIDATION.md'), 'w') as f:
        f.write('# Gazebo validation\n\n')
        f.write(f"Episodes compared: {overall['episodes']}. Collision-outcome agreement: "
                f"{overall['collision_agreement']:.1%} (criterion >= {CRITERION:.0%}: "
                f"{'MET' if passed else 'NOT MET'}).\n\n")
        f.write(f"Minimum distance: r = {overall['min_dist_r']:.3f}, mean absolute difference "
                f"{overall['min_dist_mad']:.3f} m. Same number of questions: {overall['same_asks']:.1%}.\n\n")
        f.write('## Per method and condition\n\n' + per.round(3).to_markdown(index=False) + '\n\n'
                if hasattr(per, 'to_markdown') and _has_tabulate() else
                '## Per method and condition\n\n```\n' + per.round(3).to_string(index=False) + '\n```\n\n')
        f.write('## Episodes with different collision outcomes\n\n```\n' +
                (dis.round(3).to_string(index=False) if not dis.empty else 'none') + '\n```\n')

    pd.set_option('display.width', 200)
    print(per.round(3).to_string(index=False))
    print(f"\nOverall: collision agreement {overall['collision_agreement']:.1%} "
          f"(criterion {CRITERION:.0%}: {'MET' if passed else 'NOT MET'}), "
          f"min-distance r = {overall['min_dist_r']:.3f}, MAD = {overall['min_dist_mad']:.3f} m, "
          f"same questions {overall['same_asks']:.1%}")
    if not dis.empty:
        print('\nEpisodes with different collision outcomes:\n' + dis.round(3).to_string(index=False))
    print(f'\nWritten: comparison.csv, VALIDATION.md, fig_min_dist.png in {args.runs}')


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == '__main__':
    main()

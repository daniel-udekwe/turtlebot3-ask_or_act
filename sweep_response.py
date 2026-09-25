#!/usr/bin/env python3
"""
Response-reliability sweep for the Ask-or-Act study.

Runs the offline simulator (simulate_policies.py) for several methods while
varying how reliable asking is:
  response_prob     probability that a pedestrian answers at all
  response_delay    seconds before the answer arrives
  answer_accuracy   probability that the answer is the true intent

By default one factor is varied at a time around the baseline (1.0, 1.5 s, 1.0),
which is what the figures need; --full runs every combination. Episodes run in
parallel, and each setting is saved as soon as it finishes, so an interrupted
sweep resumes where it stopped.

Outputs (in ~/ask_or_act/sim/sweep/):
  setting_*.csv     per-episode results, one file per setting
  summary.csv       per condition, method and setting, with 95% confidence intervals
  fig_<factor>.png  collisions and asks against each factor

Usage:
  python3 sweep_response.py
  python3 sweep_response.py --full --workers 16
  python3 sweep_response.py --plots-only
"""
import argparse
import glob
import itertools
import math
import os
from multiprocessing import Pool
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import simulate_policies as sp
from generate_scenarios import CONDITIONS, ENVS, FAMILIES, build
from policy import load_predictor

HOME = os.path.expanduser('~')
BASE = os.path.join(HOME, 'ask_or_act')
OUT = os.path.join(BASE, 'sim', 'sweep')
BASELINE = dict(response_prob=1.0, response_delay=1.5, answer_accuracy=1.0)
LEVELS = dict(response_prob=[1.0, 0.8, 0.5], response_delay=[0.5, 1.5, 3.0], answer_accuracy=[1.0, 0.9])
FACTOR_LABELS = dict(response_prob='probability of an answer', response_delay='answer delay (s)',
                     answer_accuracy='answer accuracy')
METHODS = ['always_ask', 'always_cautious', 'threshold:raw', 'threshold:isotonic@stale',
           'threshold:isotonic@recal', 'threshold:isotonic_dw@recal']

_W = {}   # per-worker state: loaded predictors and policy settings


def _init(model_file, model, conds, pargs):
    for c in conds:
        preds = {('raw', None): load_predictor(model_file, model, 'raw')}
        for kind in ('isotonic', 'isotonic_dw', 'temperature'):
            for cc in {'nominal', c}:
                preds[(kind, cc)] = load_predictor(model_file, model, kind, cc)
        _W[c] = preds
    _W['args'] = pargs


def _run(task):
    cond, env, family, intent, seed, setting, methods = task
    label = build(env, family, intent, seed, cond)
    rows = []
    for m in methods:
        pol = sp.make_policy(m, label, _W[cond], _W['args'], cond)
        r = sp.simulate(label, pol, CONDITIONS[cond]['noise'], setting['response_delay'],
                        setting['response_prob'], setting['answer_accuracy'])
        rows.append(dict(condition=cond, method=m, id=label['id'], family=family, intent=intent,
                         ambiguous=label['ambiguous'], **setting, **r))
    return rows


def settings(full):
    if full:
        return [dict(zip(LEVELS, v)) for v in itertools.product(*LEVELS.values())]
    out = [dict(BASELINE)]
    for f, levels in LEVELS.items():
        out += [dict(BASELINE, **{f: v}) for v in levels if v != BASELINE[f]]
    return out


def setting_file(out, s):
    return os.path.join(out, f"setting_p{s['response_prob']}_d{s['response_delay']}_a{s['answer_accuracy']}.csv")


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion k/n."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def summarize(df):
    df = df.copy()
    for col in ('collision', 'ambiguous'):
        df[col] = df[col].astype(str).str.lower().eq('true')
    rows = []
    keys = ['condition', 'method'] + list(LEVELS)
    for key, g in df.groupby(keys, sort=False):
        cross = g[g.intent == 'crossing']
        k, n = int(cross['collision'].sum()), len(cross)
        lo, hi = wilson(k, n)
        asks = g['n_asks']
        rows.append(dict(zip(keys, key), episodes=len(g),
                         collisions_on_crossing=k / n if n else float('nan'), coll_ci_low=lo, coll_ci_high=hi,
                         asks_per_episode=asks.mean(),
                         asks_ci=1.96 * asks.std(ddof=1) / math.sqrt(len(asks)) if len(asks) > 1 else 0.0,
                         mean_delay_s=g['delay'].mean(),
                         stop_s_not_crossing=g.loc[g.intent == 'not_crossing', 'stopped_time'].mean(),
                         timeout_rate=(g['outcome'] == 'timed out').mean()))
    return pd.DataFrame(rows)


def plot(summ, out):
    conds = list(dict.fromkeys(summ['condition']))
    methods = list(dict.fromkeys(summ['method']))
    for f, levels in LEVELS.items():
        # rows where every other factor is at its baseline
        mask = pd.Series(True, index=summ.index)
        for g in LEVELS:
            if g != f:
                mask &= summ[g] == BASELINE[g]
        sub = summ[mask]
        fig, axes = plt.subplots(2, len(conds), figsize=(5 * len(conds), 7), squeeze=False)
        for j, c in enumerate(conds):
            for m in methods:
                d = sub[(sub.condition == c) & (sub.method == m)].sort_values(f)
                if d.empty:
                    continue
                yerr = [d['collisions_on_crossing'] - d['coll_ci_low'], d['coll_ci_high'] - d['collisions_on_crossing']]
                axes[0, j].errorbar(d[f], d['collisions_on_crossing'], yerr=yerr, marker='o', capsize=3, label=m)
                axes[1, j].errorbar(d[f], d['asks_per_episode'], yerr=d['asks_ci'], marker='o', capsize=3, label=m)
            axes[0, j].set_title(f'{c}')
            axes[0, j].set_ylabel('collisions on crossing encounters')
            axes[1, j].set_ylabel('questions per episode')
            for i in range(2):
                axes[i, j].set_xlabel(FACTOR_LABELS[f])
                axes[i, j].set_xticks(levels)
                axes[i, j].grid(alpha=0.3)
            if f in ('response_prob', 'answer_accuracy'):
                for i in range(2):
                    axes[i, j].invert_xaxis()        # left to right: asking becomes less reliable
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(out, f'fig_{f}.png')
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f'Saved {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--full', action='store_true', help='every combination instead of one factor at a time')
    ap.add_argument('--conditions', nargs='+', default=['nominal', 'noisy'], choices=list(CONDITIONS))
    ap.add_argument('--methods', nargs='+', default=METHODS)
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--model', default='mlp')
    ap.add_argument('--model-file', default=os.path.join(BASE, 'predictor', 'intent_model.pkl'))
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--plots-only', action='store_true', help='rebuild summary and figures from saved results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.plots_only:
        pargs = SimpleNamespace(p_low=0.2, p_high=0.8, engage_dist=1.0, ask_dist=0.6, smooth_n=5, yield_timeout=8.0)
        cond_index = {c: i for i, c in enumerate(CONDITIONS)}
        todo = [s for s in settings(args.full) if not os.path.exists(setting_file(args.out, s))]
        print(f'{len(settings(args.full))} settings, {len(todo)} still to run, {args.workers} workers')
        with Pool(args.workers, initializer=_init,
                  initargs=(args.model_file, args.model, args.conditions, pargs)) as pool:
            for n, s in enumerate(todo, 1):
                tasks = []
                for c in args.conditions:
                    base = 10000 * cond_index[c] + 5000
                    tasks += [(c, e, f, i, seed, s, args.methods) for e in ENVS for f, i in FAMILIES
                              for seed in range(base, base + args.seeds)]
                rows = [r for chunk in pool.imap_unordered(_run, tasks, chunksize=4) for r in chunk]
                pd.DataFrame(rows).to_csv(setting_file(args.out, s), index=False)
                print(f"[{n}/{len(todo)}] done: answer prob {s['response_prob']}, delay {s['response_delay']} s, "
                      f"accuracy {s['answer_accuracy']} ({len(rows)} episodes)")

    files = sorted(glob.glob(os.path.join(args.out, 'setting_*.csv')))
    if not files:
        print('No results yet.')
        return
    summ = summarize(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    summ.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
    pd.set_option('display.width', 200)
    show = summ[['condition', 'method', 'response_prob', 'response_delay', 'answer_accuracy',
                 'collisions_on_crossing', 'coll_ci_low', 'coll_ci_high', 'asks_per_episode',
                 'mean_delay_s', 'timeout_rate']]
    print(show.round(3).to_string(index=False))
    plot(summ, args.out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Final evaluation for the Ask-or-Act study (see ANALYSIS_PLAN.md).

Runs every pre-specified experiment in the offline simulator on the final seeds
(7000.. in each condition block), in parallel. Each experiment setting is saved
as soon as it finishes, so an interrupted run resumes where it stopped.

Experiments:
  main      all 8 methods, all 6 conditions, baseline settings
  band      ask bands 0.1-0.9 ... 0.4-0.6 for the threshold methods (nominal, noisy)
  deadline  decision deadline 0.5 and 0.7 m for the asking methods (nominal, noisy)
  models    threshold methods with MLPs retrained with seeds 1-4, and logistic regression

Preparation (done automatically if missing):
  predictor/intent_model_seed<k>.pkl   MLPs retrained with random_state k = 1..4
  predictor/final/dataset.csv           fresh predictor test set on the final seeds (for H4)

The response-reliability grid is run separately:
  python3 sweep_response.py --full --seed-start 7000 --seeds 100 --out sim/final/response

Usage:
  python3 final_eval.py                                  # everything, final seeds
  python3 final_eval.py --only main band
  python3 final_eval.py --seed-start 5000 --seeds 3 --out sim/final_test   # quick test on development seeds
"""
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool
from types import SimpleNamespace

import pandas as pd

import simulate_policies as sp
from generate_scenarios import CONDITIONS, ENVS, FAMILIES, build, eval_seeds
from policy import load_predictor

HOME = os.path.expanduser('~')
BASE = os.path.join(HOME, 'ask_or_act')
HERE = os.path.dirname(os.path.abspath(__file__))

ALL_METHODS = ['always_act', 'always_cautious', 'always_ask', 'oracle', 'threshold:raw',
               'threshold:isotonic@stale', 'threshold:isotonic@recal', 'threshold:isotonic_dw@recal']
THRESHOLD = ['threshold:raw', 'threshold:isotonic@stale', 'threshold:isotonic@recal', 'threshold:isotonic_dw@recal']
ASKERS = ['always_ask', 'always_cautious'] + THRESHOLD
BANDS = [(0.1, 0.9), (0.15, 0.85), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
DEADLINES = [0.5, 0.7]
MODEL_SEEDS = [1, 2, 3, 4]
PRIMARY = ['nominal', 'noisy']
BASELINE = dict(p_low=0.2, p_high=0.8, ask_dist=0.6, model_file='intent_model.pkl', model_name='mlp', model='mlp_seed0')


def experiment_settings(which):
    out = []
    if 'main' in which:
        out += [dict(BASELINE, exp='main', condition=c, methods=ALL_METHODS, tag=f'main_{c}') for c in CONDITIONS]
    if 'band' in which:
        out += [dict(BASELINE, exp='band', condition=c, methods=THRESHOLD, p_low=lo, p_high=hi,
                     tag=f'band_{lo}-{hi}_{c}') for lo, hi in BANDS for c in PRIMARY]
    if 'deadline' in which:
        out += [dict(BASELINE, exp='deadline', condition=c, methods=ASKERS, ask_dist=d,
                     tag=f'deadline_{d}_{c}') for d in DEADLINES for c in PRIMARY]
    if 'models' in which:
        variants = [(f'intent_model_seed{k}.pkl', 'mlp', f'mlp_seed{k}') for k in MODEL_SEEDS]
        variants.append(('intent_model.pkl', 'logreg', 'logreg'))
        out += [dict(BASELINE, exp='models', condition=c, methods=THRESHOLD, model_file=f, model_name=n,
                     model=lab, tag=f'models_{lab}_{c}') for f, n, lab in variants for c in PRIMARY]
    return out


_W = {'cache': {}}


def _init(pred_dir):
    _W['pred_dir'] = pred_dir


def _predictors(model_file, model_name, cond):
    key = (model_file, model_name, cond)
    if key not in _W['cache']:
        path = os.path.join(_W['pred_dir'], model_file)
        preds = {('raw', None): load_predictor(path, model_name, 'raw')}
        for kind in ('isotonic', 'isotonic_dw', 'temperature'):
            for c in {'nominal', cond}:
                preds[(kind, c)] = load_predictor(path, model_name, kind, c)
        _W['cache'][key] = preds
    return _W['cache'][key]


def _run(task):
    st, env, family, intent, seed = task
    cond = st['condition']
    label = build(env, family, intent, seed, cond)
    pargs = SimpleNamespace(p_low=st['p_low'], p_high=st['p_high'], engage_dist=1.0, ask_dist=st['ask_dist'],
                            smooth_n=5, yield_timeout=8.0)
    preds = _predictors(st['model_file'], st['model_name'], cond) \
        if any(m.startswith('threshold') for m in st['methods']) else None
    rows = []
    for m in st['methods']:
        pol = sp.make_policy(m, label, preds, pargs, cond)
        r = sp.simulate(label, pol, CONDITIONS[cond]['noise'], 1.5, 1.0, 1.0)
        rows.append(dict(exp=st['exp'], tag=st['tag'], condition=cond, method=m, model=st['model'],
                         p_low=st['p_low'], p_high=st['p_high'], ask_dist=st['ask_dist'],
                         id=label['id'], family=family, intent=intent, ambiguous=label['ambiguous'], seed=seed, **r))
    return rows


def prepare(pred_dir, need_models, test_seed_start, test_seeds):
    py = sys.executable
    if need_models:
        for k in MODEL_SEEDS:
            if not os.path.exists(os.path.join(pred_dir, f'intent_model_seed{k}.pkl')):
                print(f'Training MLP with random_state {k} ...')
                subprocess.run([py, os.path.join(HERE, 'train_predictor.py'), '--dir', pred_dir,
                                '--mlp-seed', str(k), '--suffix', f'_seed{k}'],
                               check=True, stdout=subprocess.DEVNULL)
    final_dir = os.path.join(pred_dir, 'final')
    if not os.path.exists(os.path.join(final_dir, 'dataset.csv')):
        print('Building the fresh predictor test set on the final seeds ...')
        subprocess.run([py, os.path.join(HERE, 'build_dataset.py'), '--only-test', '--out', final_dir,
                        '--test-seed-start', str(test_seed_start), '--test-seeds', str(test_seeds)],
                       check=True, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', default=['main', 'band', 'deadline', 'models'],
                    choices=['main', 'band', 'deadline', 'models'])
    ap.add_argument('--seed-start', type=int, default=7000)
    ap.add_argument('--seeds', type=int, default=100, help='seeds per family and condition')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--pred-dir', default=os.path.join(BASE, 'predictor'))
    ap.add_argument('--out', default=os.path.join(BASE, 'sim', 'final'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    prepare(args.pred_dir, 'models' in args.only, args.seed_start, args.seeds)
    todo = [st for st in experiment_settings(args.only)
            if not os.path.exists(os.path.join(args.out, f"{st['tag']}.csv"))]
    total = len(experiment_settings(args.only))
    print(f'{total} experiment settings, {len(todo)} still to run, {args.workers} workers, '
          f'seeds {args.seed_start}..{args.seed_start + args.seeds - 1} per condition block')
    with Pool(args.workers, initializer=_init, initargs=(args.pred_dir,)) as pool:
        for n, st in enumerate(todo, 1):
            tasks = [(st, e, f, i, s) for e in ENVS for f, i in FAMILIES
                     for s in eval_seeds(st['condition'], args.seed_start, args.seeds)]
            rows = [r for chunk in pool.imap_unordered(_run, tasks, chunksize=4) for r in chunk]
            pd.DataFrame(rows).to_csv(os.path.join(args.out, f"{st['tag']}.csv"), index=False)
            print(f"[{n}/{len(todo)}] {st['tag']}: {len(rows)} episodes")
    print(f'Done. Results in {args.out}. Next: python3 final_stats.py --results {args.out}')


if __name__ == '__main__':
    main()

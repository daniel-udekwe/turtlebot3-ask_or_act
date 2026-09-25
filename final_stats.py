#!/usr/bin/env python3
"""
Statistics for the Ask-or-Act final evaluation (see ANALYSIS_PLAN.md).

Reads the results of final_eval.py and computes:
  main table      per condition and method: collisions on crossing encounters (95% Wilson CI),
                  questions per episode (95% CI), delay, stopped time with non-crossers, ask rates
  H1, H2          McNemar exact tests on paired collisions (noisy condition)
  H3              Wilcoxon signed-rank test on questions (one-sided) and a non-inferiority
                  check on the paired collision difference (margin 2 percentage points; nominal)
  H4              decision-window ECE of isotonic vs isotonic_dw recalibration on the fresh
                  predictor test set, with scenario-bootstrap 95% CIs (every shifted condition)
  Holm-Bonferroni correction across H1a, H1b, H2, H3
  robustness      ask-band trade-off figure, deadline and predictor-seed tables

Outputs go to <results>/stats/, including RESULTS.md with every table.

Usage:
  python3 final_stats.py
  python3 final_stats.py --results ~/ask_or_act/sim/final_test
"""
import argparse
import glob
import math
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom, wilcoxon

from generate_scenarios import CONDITIONS
from intent_features import FEATURE_NAMES
from train_predictor import DECISION_WINDOW, ece

HOME = os.path.expanduser('~')
BASE = os.path.join(HOME, 'ask_or_act')
NI_MARGIN = 0.02
N_BOOT = 2000
DW = 'threshold:isotonic_dw@recal'


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def load(results, exp):
    files = sorted(glob.glob(os.path.join(results, f'{exp}_*.csv')))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for col in ('collision', 'ambiguous'):
        d[col] = d[col].astype(str).str.lower().eq('true')
    return d


def metrics(g):
    cross = g[g.intent == 'crossing']
    k, n = int(cross.collision.sum()), len(cross)
    lo, hi = wilson(k, n)
    asks = g.n_asks
    return dict(episodes=len(g), crossing=n, collisions_on_crossing=k / n if n else float('nan'),
                coll_ci_low=lo, coll_ci_high=hi, asks_per_episode=asks.mean(),
                asks_ci=1.96 * asks.std(ddof=1) / math.sqrt(len(asks)) if len(asks) > 1 else 0.0,
                ask_rate_clear=(g.loc[~g.ambiguous, 'n_asks'] > 0).mean(),
                ask_rate_ambiguous=(g.loc[g.ambiguous, 'n_asks'] > 0).mean(),
                mean_delay_s=g.delay.mean(),
                stop_s_not_crossing=g.loc[g.intent == 'not_crossing', 'stopped_time'].mean(),
                timeout_rate=(g.outcome == 'timed out').mean())


def table(d, keys):
    rows = [dict(zip(keys, k if isinstance(k, tuple) else (k,)), **metrics(g))
            for k, g in d.groupby(keys, sort=False)]
    return pd.DataFrame(rows)


def paired(d, cond, a, b, crossing_only=True):
    x = d[(d.condition == cond) & (d.method.isin([a, b]))]
    if crossing_only:
        x = x[x.intent == 'crossing']
    w = x.pivot_table(index='id', columns='method', values=['collision', 'n_asks'], aggfunc='first')
    return w


def mcnemar(ca, cb):
    """Exact two-sided McNemar test. Returns (b, c, p): b = only A collides, c = only B collides."""
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = 1.0 if b + c == 0 else min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))
    return b, c, p


def holm(pvals):
    order = np.argsort(pvals)
    adj, prev = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = prev
    return adj


def hypotheses(main):
    rng = np.random.default_rng(0)
    rows = []
    for name, a in (('H1a', 'threshold:raw'), ('H1b', 'threshold:isotonic@stale'), ('H2', 'threshold:isotonic@recal')):
        w = paired(main, 'noisy', a, DW)
        if w.empty:
            continue
        ca, cb = w['collision'][a].to_numpy(bool), w['collision'][DW].to_numpy(bool)
        b, c, p = mcnemar(ca, cb)
        rows.append(dict(hypothesis=name, condition='noisy', comparison=f'{a} vs {DW}',
                         rate_a=ca.mean(), rate_b=cb.mean(), n_pairs=len(ca), only_a=b, only_b=c, p=p,
                         direction_as_expected=ca.mean() > cb.mean()))
    # H3: questions (all episodes) and non-inferiority on collisions (crossing episodes)
    wq = paired(main, 'nominal', DW, 'always_ask', crossing_only=False)
    if not wq.empty:
        qa, qb = wq['n_asks'][DW].to_numpy(float), wq['n_asks']['always_ask'].to_numpy(float)
        diff = qa - qb
        p = 1.0 if np.all(diff == 0) else float(wilcoxon(qa, qb, alternative='less').pvalue)
        wc = paired(main, 'nominal', DW, 'always_ask')
        da = wc['collision'][DW].to_numpy(float) - wc['collision']['always_ask'].to_numpy(float)
        boots = [rng.choice(da, len(da)).mean() for _ in range(N_BOOT)]
        upper = float(np.percentile(boots, 95))
        rows.append(dict(hypothesis='H3', condition='nominal', comparison=f'{DW} vs always_ask',
                         rate_a=qa.mean(), rate_b=qb.mean(), n_pairs=len(qa), only_a=np.nan, only_b=np.nan, p=p,
                         direction_as_expected=qa.mean() < qb.mean(),
                         collision_diff=float(da.mean()), collision_diff_upper95=upper,
                         non_inferior=upper < NI_MARGIN))
    h = pd.DataFrame(rows)
    if not h.empty:
        h['p_holm'] = holm(h['p'].tolist())
        h['significant'] = h['p_holm'] < 0.05
    return h


def h4(pred_dir):
    path = os.path.join(pred_dir, 'final', 'dataset.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    d = pd.read_csv(path)
    saved = pickle.load(open(os.path.join(pred_dir, 'intent_model.pkl'), 'rb'))
    entry = saved['models']['mlp']
    rng = np.random.default_rng(0)
    rows = []
    for cond in CONDITIONS:
        t = d[(d.split == f'test@{cond}') & (d.time_to_commit <= DECISION_WINDOW)]
        if t.empty:
            continue
        p_raw = entry['model'].predict_proba(t[FEATURE_NAMES].to_numpy())[:, 1]
        y = t.label.to_numpy()
        cals = entry['calibrators'][cond]
        p_iso, p_dw = cals['isotonic'].transform(p_raw), cals['isotonic_dw'].transform(p_raw)
        e_iso, e_dw = ece(y, p_iso), ece(y, p_dw)
        scen = t.scenario.to_numpy()
        ids = np.unique(scen)
        idx = {s: np.where(scen == s)[0] for s in ids}
        diffs = []
        for _ in range(N_BOOT // 4):
            pick = np.concatenate([idx[s] for s in rng.choice(ids, len(ids))])
            diffs.append(ece(y[pick], p_iso[pick]) - ece(y[pick], p_dw[pick]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append(dict(condition=cond, samples=len(t), ece_isotonic=e_iso, ece_isotonic_dw=e_dw,
                         improvement=e_iso - e_dw, ci_low=lo, ci_high=hi, dw_better=lo > 0))
    return pd.DataFrame(rows)


def band_figure(band, main, out):
    if band.empty:
        return
    t = table(band, ['condition', 'method', 'p_low', 'p_high'])
    conds = list(dict.fromkeys(t.condition))
    fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 5), squeeze=False)
    for ax, c in zip(axes[0], conds):
        for m in dict.fromkeys(t.method):
            s = t[(t.condition == c) & (t.method == m)].sort_values('p_low')
            ax.plot(s.asks_per_episode, s.collisions_on_crossing, 'o-', label=m)
            for _, r in s.iterrows():
                ax.annotate(f"{r.p_low:g}-{r.p_high:g}", (r.asks_per_episode, r.collisions_on_crossing),
                            fontsize=6, xytext=(3, 3), textcoords='offset points')
        ref = main[(main.condition == c) & (main.method == 'always_ask')]
        if not ref.empty:
            m = metrics(ref)
            ax.plot(m['asks_per_episode'], m['collisions_on_crossing'], 'k*', ms=12, label='always_ask')
        ax.set_title(f'{c}: ask-band trade-off')
        ax.set_xlabel('questions per episode')
        ax.set_ylabel('collisions on crossing encounters')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_band_tradeoff.png'), dpi=130)
    plt.close(fig)
    t.to_csv(os.path.join(out, 'band.csv'), index=False)


def md(df, floatfmt=3):
    if df.empty:
        return '_no results_\n'
    cols = list(df.columns)
    fmt = lambda v: (f'{v:.{floatfmt}f}' if isinstance(v, (float, np.floating)) and not pd.isna(v) else str(v))
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=os.path.join(BASE, 'sim', 'final'))
    ap.add_argument('--pred-dir', default=os.path.join(BASE, 'predictor'))
    args = ap.parse_args()
    out = os.path.join(args.results, 'stats')
    os.makedirs(out, exist_ok=True)

    main_d = load(args.results, 'main')
    main_t = table(main_d, ['condition', 'method']) if not main_d.empty else pd.DataFrame()
    hyp = hypotheses(main_d) if not main_d.empty else pd.DataFrame()
    h4_t = h4(args.pred_dir)
    band_d, dl_d, mod_d = load(args.results, 'band'), load(args.results, 'deadline'), load(args.results, 'models')
    band_figure(band_d, main_d, out)
    dl_t = table(dl_d, ['condition', 'ask_dist', 'method']) if not dl_d.empty else pd.DataFrame()
    mod_t = table(mod_d, ['condition', 'model', 'method']) if not mod_d.empty else pd.DataFrame()
    mod_check = []
    for (c, m), g in mod_d.groupby(['condition', 'model']) if not mod_d.empty else []:
        r = {k: metrics(g[g.method == k])['collisions_on_crossing'] for k in g.method.unique()}
        mod_check.append(dict(condition=c, model=m, raw=r.get('threshold:raw'), stale=r.get('threshold:isotonic@stale'),
                              recal=r.get('threshold:isotonic@recal'), dw_recal=r.get(DW),
                              dw_lowest=r.get(DW) == min(r.values())))
    mod_check = pd.DataFrame(mod_check)

    for name, df in (('main_table', main_t), ('hypotheses', hyp), ('h4_predictor', h4_t),
                     ('deadline', dl_t), ('models', mod_t), ('models_check', mod_check)):
        if not df.empty:
            df.to_csv(os.path.join(out, f'{name}.csv'), index=False)

    short = ['condition', 'method', 'collisions_on_crossing', 'coll_ci_low', 'coll_ci_high',
             'asks_per_episode', 'mean_delay_s', 'stop_s_not_crossing', 'timeout_rate']
    with open(os.path.join(out, 'RESULTS.md'), 'w') as f:
        f.write('# Ask-or-Act final evaluation results\n\nGenerated by final_stats.py; see ANALYSIS_PLAN.md.\n\n')
        f.write('## Main results\n\n' + md(main_t[short] if not main_t.empty else main_t) + '\n')
        f.write('## Hypothesis tests (Holm-corrected)\n\n' + md(hyp) + '\n')
        f.write('## H4: decision-window ECE on the fresh predictor test set\n\n' + md(h4_t) + '\n')
        f.write('## Robustness: predictor seeds (collisions on crossing encounters)\n\n' + md(mod_check) + '\n')
        f.write('## Robustness: decision deadline\n\n' +
                md(dl_t[['condition', 'ask_dist', 'method', 'collisions_on_crossing', 'asks_per_episode',
                         'mean_delay_s']] if not dl_t.empty else dl_t) + '\n')
        f.write('## Robustness: ask band\n\nSee fig_band_tradeoff.png and band.csv.\n')

    pd.set_option('display.width', 200)
    if not hyp.empty:
        print('Hypothesis tests:\n' + hyp.round(4).to_string(index=False))
    if not h4_t.empty:
        print('\nH4 (decision-window ECE):\n' + h4_t.round(4).to_string(index=False))
    if not mod_check.empty:
        print('\nPredictor-seed check:\n' + mod_check.round(3).to_string(index=False))
    print(f'\nAll tables and RESULTS.md written to {out}')


if __name__ == '__main__':
    main()

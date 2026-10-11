#!/usr/bin/env python3
"""
Paper figures and tables for the Ask-or-Act study.

Generates every figure and table from the frozen final results, with one style:
the same colour and name for each method everywhere, colour-blind-safe colours
(Okabe-Ito), journal-sized fonts, and PDF (vector) plus PNG output.

Inputs (defaults under ~/ask_or_act):
  predictor/intent_model.pkl, predictor/final/dataset.csv   fresh predictor test set
  sim/final/main_*.csv, band_*.csv, models_*.csv             final_eval.py
  sim/final/stats/hypotheses.csv                             final_stats.py
  sim/final/response/summary.csv                             sweep_response.py --full
  runs/validation/comparison.csv                             compare_validation.py
  scenarios: generated on the fly from the final seeds

Outputs (in ~/ask_or_act/paper/):
  figures/fig_scenarios, fig_ece_hidden, fig_reliability, fig_collisions_conditions,
          fig_band_tradeoff, fig_response, fig_validation   (.pdf and .png)
  tables/tab_calibration, tab_main, tab_hypotheses, tab_robustness, tab_validation   (.tex)

Usage:
  python3 make_figures.py
  python3 make_figures.py --only fig_band_tradeoff tab_main
"""
import argparse
import math
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from final_stats import load, metrics, wilson
from generate_scenarios import CONDITIONS, FAMILIES, ROBOT, build, eval_seeds
from intent_features import FEATURE_NAMES
from ped_traj import build_trajectory
from train_predictor import DECISION_WINDOW, ece, reliability

HOME = os.path.expanduser('~')
BASE = os.path.join(HOME, 'ask_or_act')

# ---------------------------------------------------------------- style
OKABE = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73', yellow='#F0E442',
             blue='#0072B2', vermillion='#D55E00', purple='#CC79A7', grey='#999999')
METHOD = {  # key: (display name, colour, marker)
    'always_act': ('Always act', OKABE['grey'], 'x'),
    'always_cautious': ('Always cautious', OKABE['green'], 'v'),
    'always_ask': ('Always ask', OKABE['black'], '*'),
    'oracle': ('Oracle', OKABE['yellow'], 'P'),
    'threshold:raw': ('Raw', OKABE['purple'], 'o'),
    'threshold:isotonic@stale': ('Isotonic (stale)', OKABE['sky'], 's'),
    'threshold:isotonic@recal': ('Isotonic (recalibrated)', OKABE['blue'], 'D'),
    'threshold:isotonic_dw@recal': ('Decision-aware (recalibrated)', OKABE['vermillion'], '^'),
}
CAL = {  # predictor calibration variants for the calibration figures
    'raw': ('Raw', OKABE['purple']),
    'isotonic': ('Isotonic (recalibrated)', OKABE['blue']),
    'isotonic_dw': ('Decision-aware (recalibrated)', OKABE['vermillion']),
}
COND_NAME = dict(nominal='Nominal', slow='Slow', fast='Fast', long_pause='Long pause', noisy='Noisy',
                 combined='Combined')
FAMILY_NAME = dict(clear_cross='Clear crossing', clear_noncross='Clear non-crossing', hesitant='Hesitant',
                   late_turn='Late turn', slow_cross='Slow crossing', wait_far='Waits far back')
COL1, COL2 = 3.5, 7.16      # single- and double-column widths (inches)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.dpi': 150, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


def save(fig, out, name):
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, 'figures', f'{name}.{ext}'), dpi=300)
    plt.close(fig)
    print(f'  figures/{name}.pdf')


def write_tex(rows, header, out, name, caption, label, colfmt):
    os.makedirs(os.path.join(out, 'tables'), exist_ok=True)
    lines = [r'\begin{table}[t]', r'\centering', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             rf'\begin{{tabular}}{{{colfmt}}}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    for r in rows:
        lines.append(r'\midrule' if r == 'MIDRULE' else ' & '.join(str(v) for v in r) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(os.path.join(out, 'tables', f'{name}.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  tables/{name}.tex')


def pct(x, d=1):
    return '--' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{100 * x:.{d}f}'


# ------------------------------------------------------------ predictor data
def predictor_eval(pred_dir):
    """Per condition: arrays of labels, window mask and calibrated probabilities on the fresh test set."""
    path = os.path.join(pred_dir, 'final', 'dataset.csv')
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    entry = pickle.load(open(os.path.join(pred_dir, 'intent_model.pkl'), 'rb'))['models']['mlp']
    out = {}
    for c in CONDITIONS:
        t = d[d.split == f'test@{c}']
        if t.empty:
            continue
        p = entry['model'].predict_proba(t[FEATURE_NAMES].to_numpy())[:, 1]
        cal = entry['calibrators'][c]
        out[c] = dict(y=t.label.to_numpy(), win=t.time_to_commit.to_numpy() <= DECISION_WINDOW,
                      p={'raw': p, 'isotonic': cal['isotonic'].transform(p), 'isotonic_dw': cal['isotonic_dw'].transform(p)})
    return out


# ------------------------------------------------------------------ figures
def fig_scenarios(out, maps_dir):
    """(a) routes of each family on the cafe map; (b) signed distance to the robot's path over time."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL2, 3.4), gridspec_kw=dict(width_ratios=[1, 1.4]))
    pgm, yml = os.path.join(maps_dir, 'cafe.pgm'), os.path.join(maps_dir, 'cafe.yaml')
    if os.path.exists(pgm):
        from plot_map import read_map_yaml, read_pgm
        img = read_pgm(pgm)
        res, origin = read_map_yaml(yml)
        h, w = img.shape
        ax.imshow(img, cmap='gray', extent=[origin[0], origin[0] + w * res, origin[1], origin[1] + h * res],
                  origin='upper', alpha=0.5)
    ax.plot([ROBOT['start'][0], ROBOT['goal'][0]], [ROBOT['start'][1], ROBOT['goal'][1]], color=OKABE['black'], lw=2)
    ax.annotate('', xy=(ROBOT['goal'][0], ROBOT['goal'][1]), xytext=(ROBOT['goal'][0] + 0.6, ROBOT['goal'][1]),
                arrowprops=dict(arrowstyle='->', color=OKABE['black']))
    ax.annotate('robot path', (0.0, 0.05), xytext=(0, 5), textcoords='offset points', fontsize=7, ha='center')
    cols = [OKABE[k] for k in ('orange', 'sky', 'green', 'blue', 'vermillion', 'purple')]
    fams = list(dict.fromkeys(f for f, _ in FAMILIES))
    fam_col = {f: cols[i] for i, f in enumerate(fams)}
    handles = []
    for fam, intent in FAMILIES:
        lab = build('cafe', fam, intent, eval_seeds('nominal', 7000, 1)[0], 'nominal')
        dx = 0.12 * (fams.index(fam) - (len(fams) - 1) / 2)          # display offset so routes do not overlap
        ls = '-' if intent == 'crossing' else '--'
        pts = [lab['start']] + lab['waypoints']
        ax.plot([p[0] - lab['route_shift'] + dx for p in pts], [p[1] for p in pts], ls, color=fam_col[fam], lw=1.2)
        if fam in ('hesitant', 'wait_far'):                           # where the pedestrian stops
            stop = lab['waypoints'][0]
            ax.plot(stop[0] - lab['route_shift'] + dx, stop[1], 'o', ms=3, color=fam_col[fam])
        traj = build_trajectory(lab['start'], lab['plan'], lab['human_speed'])
        t = np.array([q[0] for q in traj])
        dist = np.array([-q[2] for q in traj])                        # > 0 before the line, < 0 once across
        keep = t <= 10
        h, = bx.plot(t[keep], dist[keep], ls, color=fam_col[fam], lw=1.2,
                     label=f"{FAMILY_NAME[fam]} ({'crosses' if intent == 'crossing' else 'no'})")
        handles.append(h)
    ax.plot([], [], 'o', ms=3, color=OKABE['grey'], label='stops here')
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-3.8, 4.0)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('(a) Routes (solid: crosses; dashed: does not)')
    bx.axhline(0, color=OKABE['black'], lw=0.8)
    bx.text(9.9, 0.08, "robot's path", fontsize=6, ha='right')
    bx.set_xlim(0, 10)
    bx.set_ylim(-1.2, 3.4)
    bx.set_xlabel('time since release (s)')
    bx.set_ylabel("distance to the robot's path (m)")
    bx.set_title('(b) Approach profiles (ambiguous pairs coincide until they diverge)')
    bx.grid(alpha=0.3)
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out, 'fig_scenarios')


def fig_ece_hidden(out, pe):
    """Aggregate vs decision-window ECE per condition: the miscalibration aggregate metrics hide."""
    conds = [c for c in CONDITIONS if c in pe]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4), sharey=True)
    width = 0.26
    for ax, (title, use_win) in zip(axes, (('(a) All test samples', False), ('(b) Decision window only', True))):
        for k, (key, (name, col)) in enumerate(CAL.items()):
            vals = []
            for c in conds:
                m = pe[c]['win'] if use_win else np.ones(len(pe[c]['y']), bool)
                vals.append(ece(pe[c]['y'][m], pe[c]['p'][key][m]))
            ax.bar(np.arange(len(conds)) + (k - 1) * width, vals, width, color=col, label=name)
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels([COND_NAME[c] for c in conds], rotation=30, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('expected calibration error')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    save(fig, out, 'fig_ece_hidden')


def fig_reliability(out, pe):
    rows = [c for c in ('nominal', 'noisy') if c in pe]
    fig, axes = plt.subplots(len(rows), 2, figsize=(COL2 * 0.75, 2.6 * len(rows)), squeeze=False)
    for i, c in enumerate(rows):
        for j, (title, use_win) in enumerate((('all samples', False), ('decision window', True))):
            ax = axes[i, j]
            m = pe[c]['win'] if use_win else np.ones(len(pe[c]['y']), bool)
            ax.plot([0, 1], [0, 1], ':', color=OKABE['grey'], lw=1)
            for key, (name, col) in CAL.items():
                conf, acc, _ = reliability(pe[c]['y'][m], pe[c]['p'][key][m])
                ax.plot(conf, acc, 'o-', ms=3, color=col, label=name)
            ax.set_title(f'{COND_NAME[c]}, {title}')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xlabel('predicted P(crossing)')
            ax.set_ylabel('observed frequency')
            ax.grid(alpha=0.3)
    fig.tight_layout(h_pad=2.5, w_pad=1.5)    # space between rows so x-labels do not touch the titles below
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, out, 'fig_reliability')


def fig_collisions_conditions(out, main):
    methods = ['always_cautious', 'threshold:raw', 'threshold:isotonic@stale', 'threshold:isotonic@recal',
               'threshold:isotonic_dw@recal', 'always_ask']
    conds = [c for c in CONDITIONS if c in set(main.condition)]
    fig, ax = plt.subplots(figsize=(COL2, 2.4))
    width = 0.8 / len(methods)
    for k, m in enumerate(methods):
        vals, lo, hi = [], [], []
        for c in conds:
            r = metrics(main[(main.condition == c) & (main.method == m)])
            vals.append(r['collisions_on_crossing'])
            lo.append(r['collisions_on_crossing'] - r['coll_ci_low'])
            hi.append(r['coll_ci_high'] - r['collisions_on_crossing'])
        name, col, _ = METHOD[m]
        ax.bar(np.arange(len(conds)) + (k - (len(methods) - 1) / 2) * width, vals, width, yerr=[lo, hi],
               color=col, label=name, capsize=1.5, error_kw=dict(lw=0.6))
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels([COND_NAME[c] for c in conds])
    ax.set_ylabel('collisions on crossing encounters')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(ncol=3, fontsize=6, loc='upper left')
    save(fig, out, 'fig_collisions_conditions')


def fig_band_tradeoff(out, band, main):
    methods = ['threshold:raw', 'threshold:isotonic@stale', 'threshold:isotonic@recal', 'threshold:isotonic_dw@recal']
    conds = [c for c in ('nominal', 'noisy') if c in set(band.condition)]
    fig, axes = plt.subplots(1, len(conds), figsize=(COL2, 2.7), squeeze=False)
    for ax, c in zip(axes[0], conds):
        for m in methods:
            pts = []
            for (lo, hi), g in band[(band.condition == c) & (band.method == m)].groupby(['p_low', 'p_high']):
                r = metrics(g)
                pts.append((lo, hi, r['asks_per_episode'], r['collisions_on_crossing']))
            pts.sort()
            name, col, mk = METHOD[m]
            ax.plot([p[2] for p in pts], [p[3] for p in pts], '-', marker=mk, ms=3.5, color=col, label=name)
            for lo, hi, x, y in pts:
                if (lo, hi) in ((0.4, 0.6), (0.1, 0.9)) and m == 'threshold:isotonic_dw@recal':
                    ax.annotate(f'{lo:g}–{hi:g}', (x, y), xytext=(4, 4), textcoords='offset points', fontsize=6,
                                color=col)
        ref = main[(main.condition == c) & (main.method == 'always_ask')]
        if not ref.empty:
            r = metrics(ref)
            ax.plot(r['asks_per_episode'], r['collisions_on_crossing'], '*', ms=9, color=OKABE['black'],
                    label='Always ask')
        ax.set_title(f'{COND_NAME[c]}')
        ax.set_xlabel('questions per episode')
        ax.set_ylabel('collisions on crossing encounters')
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    save(fig, out, 'fig_band_tradeoff')


def fig_response(out, resp):
    methods = ['always_ask', 'threshold:raw', 'threshold:isotonic_dw@recal']
    conds = [c for c in ('nominal', 'noisy') if c in set(resp.condition)]
    r = resp[resp.answer_accuracy == 0.9]
    styles = {1.0: '-', 0.8: '--', 0.5: ':'}
    fig, axes = plt.subplots(1, len(conds), figsize=(COL2, 2.5), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], conds):
        for m in methods:
            name, col, mk = METHOD[m]
            for prob, ls in styles.items():
                d = r[(r.condition == c) & (r.method == m) & (r.response_prob == prob)].sort_values('response_delay')
                if d.empty:
                    continue
                ax.plot(d.response_delay, d.collisions_on_crossing, ls, marker=mk, ms=3.5, color=col,
                        label=f'{name}, {int(prob * 100)}% answer' if c == conds[0] else None)
        ax.set_title(f'{COND_NAME[c]}, 90% answer accuracy')
        ax.set_xlabel('answer delay (s)')
        ax.set_xticks(sorted(r.response_delay.unique()))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel('collisions on crossing encounters')
    axes[0, 0].legend(fontsize=5.5, ncol=1)
    save(fig, out, 'fig_response')


def fig_validation(out, comp):
    fig, ax = plt.subplots(figsize=(COL1, COL1))
    for m in dict.fromkeys(comp.method):
        g = comp[comp.method == m]
        name, col, mk = METHOD[m]
        ax.scatter(g.min_dist_offline, g.min_dist_gazebo, s=10, marker=mk, color=col, alpha=0.75, label=name)
    lim = [0, max(comp.min_dist_offline.max(), comp.min_dist_gazebo.max()) * 1.05]
    ax.plot(lim, lim, ':', color=OKABE['grey'], lw=1)
    ax.axhline(0.4, color=OKABE['vermillion'], lw=0.6, ls='--')
    ax.axvline(0.4, color=OKABE['vermillion'], lw=0.6, ls='--')
    agree = (comp.collision_gazebo == comp.collision_offline).mean()
    r = np.corrcoef(comp.min_dist_offline, comp.min_dist_gazebo)[0, 1]
    ax.set_title(f'Collision agreement {agree:.1%}, r = {r:.3f}')
    ax.set_xlabel('minimum distance, offline simulator (m)')
    ax.set_ylabel('minimum distance, Gazebo (m)')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    save(fig, out, 'fig_validation')


# ------------------------------------------------------------------- tables
def tab_calibration(out, pe):
    rows = []
    for c in [c for c in CONDITIONS if c in pe]:
        y, w = pe[c]['y'], pe[c]['win']
        row = [COND_NAME[c]]
        for key in CAL:
            p = pe[c]['p'][key]
            row += [f'{ece(y, p):.3f}', f'{ece(y[w], p[w]):.3f}']
        row += [f"{roc_auc_score(y[w], pe[c]['p']['raw'][w]):.3f}"]
        rows.append(row)
    header = ['Condition', 'Raw all', 'Raw window', 'Iso. all', 'Iso. window', 'D.-a. all', 'D.-a. window', 'AUC window']
    write_tex(rows, header, out, 'tab_calibration',
              'Expected calibration error on the fresh test set, over all samples and within the decision window '
              '(recalibrated on each condition). AUC is unaffected by calibration.',
              'tab:calibration', 'l' + 'c' * 7)


def tab_main(out, main):
    methods = ['always_act', 'always_cautious', 'always_ask', 'oracle', 'threshold:raw', 'threshold:isotonic@stale',
               'threshold:isotonic@recal', 'threshold:isotonic_dw@recal']
    rows = []
    for c in ('nominal', 'noisy'):
        if rows:
            rows.append('MIDRULE')
        for m in methods:
            g = main[(main.condition == c) & (main.method == m)]
            if g.empty:
                continue
            r = metrics(g)
            rows.append([COND_NAME[c] if m == methods[0] else '', METHOD[m][0],
                         f"{pct(r['collisions_on_crossing'])} [{pct(r['coll_ci_low'])}, {pct(r['coll_ci_high'])}]",
                         f"{r['asks_per_episode']:.2f}", f"{r['mean_delay_s']:.2f}", f"{r['stop_s_not_crossing']:.2f}"])
    write_tex(rows, ['Condition', 'Method', r'Collisions (\%) [95\% CI]', 'Questions/ep.', 'Delay (s)',
                     'Stop, non-crossers (s)'], out, 'tab_main',
              'Closed-loop results on the final scenarios (400 crossing encounters, 800 episodes per method and condition).',
              'tab:main', 'llcccc')


def tab_hypotheses(out, hyp):
    rows = []
    for _, h in hyp.iterrows():
        if h.hypothesis == 'H3':
            detail = (f"questions {h.rate_a:.2f} vs {h.rate_b:.2f}; collision difference "
                      f"{100 * h.collision_diff:+.1f} pp (upper 95\\% {100 * h.collision_diff_upper95:.1f} pp)")
        else:
            detail = f"collisions {pct(h.rate_a)}\\% vs {pct(h.rate_b)}\\%; discordant {int(h.only_a)} vs {int(h.only_b)}"
        p = '<0.001' if h.p_holm < 0.001 else f'{h.p_holm:.3f}'
        rows.append([h.hypothesis, COND_NAME.get(h.condition, h.condition), detail, p,
                     'yes' if h.significant and h.direction_as_expected else 'no'])
    write_tex(rows, ['', 'Condition', 'Result', r'$p$ (Holm)', 'Supported'], out, 'tab_hypotheses',
              'Pre-specified hypothesis tests (McNemar exact for H1--H2; Wilcoxon signed-rank and non-inferiority, '
              'margin 2 percentage points, for H3).', 'tab:hypotheses', 'llp{8cm}cc')


def tab_robustness(out, main, models):
    methods = ['threshold:raw', 'threshold:isotonic@stale', 'threshold:isotonic@recal', 'threshold:isotonic_dw@recal']
    rows = []
    for c in ('nominal', 'noisy'):
        row = [COND_NAME[c]]
        for m in methods:
            vals = [metrics(main[(main.condition == c) & (main.method == m)])['collisions_on_crossing']]
            for mod, g in models[(models.condition == c) & (models.method == m)].groupby('model'):
                if mod.startswith('mlp'):
                    vals.append(metrics(g)['collisions_on_crossing'])
            lr = models[(models.condition == c) & (models.method == m) & (models.model == 'logreg')]
            row.append(f"{pct(min(vals))}--{pct(max(vals))} ({pct(metrics(lr)['collisions_on_crossing']) if not lr.empty else '--'})")
        rows.append(row)
    write_tex(rows, ['Condition'] + [METHOD[m][0] for m in methods], out, 'tab_robustness',
              r'Collisions on crossing encounters (\%): range over five neural-network predictors trained with '
              'different seeds; logistic regression in parentheses.', 'tab:robustness', 'lcccc')


def tab_validation(out, comp):
    rows = []
    for (m, c), g in comp.groupby(['method', 'condition'], sort=False):
        rows.append([METHOD[m][0], COND_NAME[c], pct((g.collision_gazebo == g.collision_offline).mean()),
                     pct(g.collision_gazebo.mean()), pct(g.collision_offline.mean()),
                     f'{np.corrcoef(g.min_dist_gazebo, g.min_dist_offline)[0, 1]:.3f}',
                     f'{(g.min_dist_gazebo - g.min_dist_offline).median() * 100:+.1f}'])
    write_tex(rows, ['Method', 'Condition', r'Agreement (\%)', r'Coll. Gazebo (\%)', r'Coll. offline (\%)',
                     r'$r$ (min. dist.)', 'Median diff. (cm)'], out, 'tab_validation',
              'Gazebo validation: 40 episodes per method and condition, compared one to one with the offline simulator.',
              'tab:validation', 'llccccc')


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred-dir', default=os.path.join(BASE, 'predictor'))
    ap.add_argument('--results', default=os.path.join(BASE, 'sim', 'final'))
    ap.add_argument('--response', default=os.path.join(BASE, 'sim', 'final', 'response', 'summary.csv'))
    ap.add_argument('--validation', default=os.path.join(BASE, 'runs', 'validation', 'comparison.csv'))
    ap.add_argument('--maps', default=os.path.join(HOME, 'hunav_ws', 'src', 'hunav_gazebo_wrapper', 'maps'))
    ap.add_argument('--out', default=os.path.join(BASE, 'paper'))
    ap.add_argument('--only', nargs='+', default=None, help='names of figures/tables to make')
    args = ap.parse_args()
    want = lambda n: args.only is None or n in args.only

    main_d, band, models = load(args.results, 'main'), load(args.results, 'band'), load(args.results, 'models')
    hyp_path = os.path.join(args.results, 'stats', 'hypotheses.csv')
    pe = predictor_eval(args.pred_dir) if any(want(n) for n in ('fig_ece_hidden', 'fig_reliability', 'tab_calibration')) else None
    resp = pd.read_csv(args.response) if os.path.exists(args.response) else pd.DataFrame()
    comp = pd.read_csv(args.validation) if os.path.exists(args.validation) else pd.DataFrame()
    for col in ('collision_gazebo', 'collision_offline'):
        if col in comp:
            comp[col] = comp[col].astype(str).str.lower().eq('true')

    jobs = [('fig_scenarios', True, lambda: fig_scenarios(args.out, args.maps)),
            ('fig_ece_hidden', pe, lambda: fig_ece_hidden(args.out, pe)),
            ('fig_reliability', pe, lambda: fig_reliability(args.out, pe)),
            ('fig_collisions_conditions', not main_d.empty, lambda: fig_collisions_conditions(args.out, main_d)),
            ('fig_band_tradeoff', not band.empty, lambda: fig_band_tradeoff(args.out, band, main_d)),
            ('fig_response', not resp.empty, lambda: fig_response(args.out, resp)),
            ('fig_validation', not comp.empty, lambda: fig_validation(args.out, comp)),
            ('tab_calibration', pe, lambda: tab_calibration(args.out, pe)),
            ('tab_main', not main_d.empty, lambda: tab_main(args.out, main_d)),
            ('tab_hypotheses', os.path.exists(hyp_path), lambda: tab_hypotheses(args.out, pd.read_csv(hyp_path))),
            ('tab_robustness', not models.empty, lambda: tab_robustness(args.out, main_d, models)),
            ('tab_validation', not comp.empty, lambda: tab_validation(args.out, comp))]
    print(f'Writing to {args.out}')
    for name, available, fn in jobs:
        if not want(name):
            continue
        if available is None or available is False:
            print(f'  {name}: skipped (input not found)')
            continue
        fn()


if __name__ == '__main__':
    main()

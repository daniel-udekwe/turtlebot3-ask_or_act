#!/usr/bin/env python3
"""
Train, calibrate, and evaluate intent predictors for the Ask-or-Act study.

Models:       logistic regression, small neural network (MLP)
Training:     nominal condition only
Calibration:  none (raw); temperature scaling and isotonic regression fitted on the
              nominal calibration set ("stale" under shift) or on each shifted
              condition's own calibration set ("recalibrated"); and decision-aware
              isotonic calibration ("isotonic_dw"), fitted only on samples within
              DECISION_WINDOW s of the moment a crossing pedestrian reaches the line,
              where the robot actually has to decide
Metrics:      ECE, Brier score, log loss, AUC, accuracy on each condition's test set,
              overall and for clear vs ambiguous scenarios
Outputs (in ~/ask_or_act/predictor/):
  metrics.csv              one row per model / calibration / subset
  reliability_<model>.png  reliability diagrams on the test split
  intent_model.pkl         all fitted models and calibrators, for use by the robot

Usage:
  python3 train_predictor.py
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibration import EPS, IsotonicCalibrator, TemperatureScaler
from intent_features import FEATURE_NAMES

HOME = os.path.expanduser('~')
DIR = os.path.join(HOME, 'ask_or_act', 'predictor')
DECISION_WINDOW = 3.5    # s before the commit time; covers the robot's decision deadline
def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    conf, acc, n = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 10:
            conf.append(p[m].mean())
            acc.append(y[m].mean())
            n.append(m.sum())
    return np.array(conf), np.array(acc), np.array(n)


def metrics(y, p):
    return dict(n=len(y), ece=ece(y, p), brier=brier_score_loss(y, p),
                log_loss=log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]),
                auc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                accuracy=float(((p >= 0.5) == y).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=DIR)
    args = ap.parse_args()

    d = pd.read_csv(os.path.join(args.dir, 'dataset.csv'))
    conds = [c for c in d['condition'].unique() if f'test@{c}' in set(d['split'])]
    conds = ['nominal'] + [c for c in conds if c != 'nominal']
    part = lambda split: d[d.split == split]
    xy = lambda df: (df[FEATURE_NAMES].to_numpy(), df['label'].to_numpy())

    models = {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        'mlp': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500,
                                                             early_stopping=True, random_state=0)),
    }
    X_train, Y_train = xy(part('train'))
    rows, saved = [], {'feature_names': FEATURE_NAMES, 'conditions': conds, 'models': {}}
    for name, model in models.items():
        model.fit(X_train, Y_train)
        # Calibrators fitted on each condition's calibration set
        cals = {}
        for c in conds:
            Xc, Yc = xy(part(f'calib@{c}'))
            pc = model.predict_proba(Xc)[:, 1]
            win = part(f'calib@{c}')['time_to_commit'].to_numpy() <= DECISION_WINDOW
            cals[c] = {'temperature': TemperatureScaler().fit(pc, Yc), 'isotonic': IsotonicCalibrator().fit(pc, Yc),
                       'isotonic_dw': IsotonicCalibrator().fit(pc[win], Yc[win])}
        saved['models'][name] = dict(model=model, calibrators=cals)

        fig, axes = plt.subplots(2, (len(conds) + 1) // 2, figsize=(4 * ((len(conds) + 1) // 2), 8), squeeze=False)
        for ax, c in zip(axes.flat, conds):
            test = part(f'test@{c}')
            Xt, Yt = xy(test)
            amb = test['ambiguous'].astype(str).str.lower().eq('true').to_numpy()
            dwin = test['time_to_commit'].to_numpy() <= DECISION_WINDOW
            p_raw = model.predict_proba(Xt)[:, 1]
            variants = {'raw': p_raw}
            for kind in ('temperature', 'isotonic', 'isotonic_dw'):
                variants[f'{kind} (nominal calib)'] = cals['nominal'][kind].transform(p_raw)
                if c != 'nominal':
                    variants[f'{kind} (recalibrated)'] = cals[c][kind].transform(p_raw)
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            for vname, p in variants.items():
                for subset, mask in (('all', np.ones(len(Yt), bool)), ('clear', ~amb), ('ambiguous', amb),
                                     ('decision_window', dwin)):
                    rows.append(dict(model=name, condition=c, calibration=vname, subset=subset,
                                     **metrics(Yt[mask], p[mask])))
                if vname in ('raw', 'isotonic (nominal calib)', 'isotonic (recalibrated)'):
                    conf, acc, _ = reliability(Yt, p)
                    ax.plot(conf, acc, 'o-', ms=4, label=f'{vname}: ECE {ece(Yt, p):.3f}')
            ax.set_title(f'{name}, test: {c}')
            ax.set_xlabel('predicted P(crossing)')
            ax.set_ylabel('observed frequency')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        for ax in list(axes.flat)[len(conds):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(args.dir, f'reliability_{name}.png'), dpi=110)
        plt.close(fig)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.dir, 'metrics.csv'), index=False)
    with open(os.path.join(args.dir, 'intent_model.pkl'), 'wb') as f:
        pickle.dump(saved, f)

    pd.set_option('display.width', 160)
    for metric in ('ece', 'brier'):
        for subset in ('all', 'decision_window'):
            tab = res[res.subset == subset].pivot_table(index=['model', 'calibration'], columns='condition',
                                                        values=metric)[conds]
            print(f'\n{metric.upper()} on the test set, {subset} scenarios (lower is better)')
            print(tab.round(3).to_string())
    print(f"\nSaved metrics.csv, reliability plots and intent_model.pkl in {args.dir}")


if __name__ == '__main__':
    main()

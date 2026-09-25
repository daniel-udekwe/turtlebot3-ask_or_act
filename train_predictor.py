#!/usr/bin/env python3
"""
Train, calibrate, and evaluate intent predictors for the Ask-or-Act study.

Models:       logistic regression, small neural network (MLP)
Calibration:  none (raw), temperature scaling, isotonic regression (fit on the calib split)
Metrics:      expected calibration error (ECE), Brier score, log loss, AUC, accuracy,
              on the test split, overall and for clear vs ambiguous scenarios
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
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from intent_features import FEATURE_NAMES

HOME = os.path.expanduser('~')
DIR = os.path.join(HOME, 'ask_or_act', 'predictor')
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureScaler:
    """Divides the model's logit by a single temperature T fitted to minimise log loss."""
    def fit(self, p, y):
        z = logit(p)
        res = minimize_scalar(lambda lt: log_loss(y, np.clip(sigmoid(z / np.exp(lt)), EPS, 1 - EPS)),
                              bounds=(-3, 3), method='bounded')
        self.T = float(np.exp(res.x))
        return self

    def transform(self, p):
        return sigmoid(logit(p) / self.T)


class IsotonicCalibrator:
    def fit(self, p, y):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(p, y)
        return self

    def transform(self, p):
        return np.clip(self.iso.predict(p), EPS, 1 - EPS)


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
    parts = {s: d[d.split == s] for s in ('train', 'calib', 'test')}
    X = {s: v[FEATURE_NAMES].to_numpy() for s, v in parts.items()}
    Y = {s: v['label'].to_numpy() for s, v in parts.items()}
    amb_test = parts['test']['ambiguous'].astype(str).str.lower().eq('true').to_numpy()

    models = {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        'mlp': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500,
                                                             early_stopping=True, random_state=0)),
    }
    rows, saved = [], {'feature_names': FEATURE_NAMES, 'models': {}}
    for name, model in models.items():
        model.fit(X['train'], Y['train'])
        p_cal = model.predict_proba(X['calib'])[:, 1]
        p_test_raw = model.predict_proba(X['test'])[:, 1]
        calibrators = {'raw': None,
                       'temperature': TemperatureScaler().fit(p_cal, Y['calib']),
                       'isotonic': IsotonicCalibrator().fit(p_cal, Y['calib'])}
        saved['models'][name] = dict(model=model, calibrators=calibrators)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect calibration')
        for cname, cal in calibrators.items():
            p = p_test_raw if cal is None else cal.transform(p_test_raw)
            for subset, mask in (('all', slice(None)), ('clear', ~amb_test), ('ambiguous', amb_test)):
                m = metrics(Y['test'][mask], p[mask])
                rows.append(dict(model=name, calibration=cname, subset=subset, **m))
            conf, acc, _ = reliability(Y['test'], p)
            ax.plot(conf, acc, 'o-', label=f"{cname} (ECE {ece(Y['test'], p):.3f})")
        if calibrators['temperature'] is not None:
            ax.set_title(f"{name}: test reliability (T = {calibrators['temperature'].T:.2f})")
        ax.set_xlabel('predicted probability of crossing')
        ax.set_ylabel('observed frequency of crossing')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.dir, f'reliability_{name}.png'), dpi=120)
        plt.close(fig)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.dir, 'metrics.csv'), index=False)
    with open(os.path.join(args.dir, 'intent_model.pkl'), 'wb') as f:
        pickle.dump(saved, f)

    pd.set_option('display.width', 120)
    for subset in ('all', 'clear', 'ambiguous'):
        print(f'\nTest set, {subset} scenarios')
        print(res[res.subset == subset].drop(columns='subset').round(3).to_string(index=False))
    print(f"\nSaved metrics.csv, reliability plots and intent_model.pkl in {args.dir}")


if __name__ == '__main__':
    main()

"""
Ask-or-act decision policies for the Ask-or-Act study.

Pure Python (no ROS), so the Gazebo controller (line_controller.py) and the
offline simulator (simulate_policies.py) run exactly the same decision logic.

Each step the policy looks at every pedestrian that is engaged (ahead of the
robot and within `engage_dist` along the robot's line, and not yet past it) and
returns GO or STOP. `ask_dist` (< engage_dist) is the decision deadline: an
uncertain robot keeps driving and observing until the pedestrian is this close
ahead, and only then asks (or, for always_cautious, stops).

  always_act      never reacts (baseline)
  always_cautious at the deadline, stops if the pedestrian is within `caution_dist` of the line
  always_ask      at the deadline, asks once, stops until answered, then follows the answer
  threshold       predicts P(crossing) with the (calibrated) intent model, averaged over
                  the last `smooth_n` predictions (so noise cannot trigger a decision):
                    p >= p_high -> stop (yield);  p <= p_low -> go;
                    otherwise keep going until the deadline, then ask once and stop
                    until answered, then follow the answer
  oracle          knows the true intent and yields only to crossing pedestrians (upper bound)
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intent_features import compute_features   # noqa: E402

POLICIES = ('always_act', 'always_cautious', 'always_ask', 'threshold', 'oracle')
HISTORY = 40            # observations kept per pedestrian (4 s at 10 Hz)


def load_predictor(model_file, model_name='mlp', calibration='isotonic', calib_condition='nominal'):
    """Return (model, calibrator or None) from intent_model.pkl."""
    saved = pickle.load(open(model_file, 'rb'))
    entry = saved['models'][model_name]
    cal = None if calibration == 'raw' else entry['calibrators'][calib_condition][calibration]
    return entry['model'], cal


class AskOrActPolicy:
    def __init__(self, policy, model=None, calibrator=None, p_low=0.2, p_high=0.8,
                 engage_dist=1.0, ask_dist=0.6, clear_dist=0.6, caution_dist=1.5, answer_timeout=5.0,
                 true_intent='', smooth_n=5):
        if policy not in POLICIES:
            raise ValueError(f'unknown policy {policy}; choose from {POLICIES}')
        if policy == 'threshold' and model is None:
            raise ValueError('the threshold policy needs a model')
        self.policy, self.model, self.calibrator = policy, model, calibrator
        self.p_low, self.p_high = p_low, p_high
        self.engage_dist, self.ask_dist = engage_dist, ask_dist
        self.clear_dist, self.caution_dist = clear_dist, caution_dist
        self.answer_timeout, self.true_intent = answer_timeout, true_intent
        self.smooth_n = max(1, int(smooth_n))
        self.p_hist = {}
        self.obs, self.side0, self.cleared = {}, {}, set()
        self.asked, self.knowledge, self.p_cache = {}, {}, {}
        self.asks, self.answers, self.decisions = [], [], []

    def add_observation(self, hid, t, x, y):
        ob = self.obs.setdefault(hid, [])
        ob.append((t, x, y))
        del ob[:-HISTORY]
        self.side0.setdefault(hid, 1.0 if y >= 0 else -1.0)

    def receive_answer(self, hid, intent, t):
        if hid in self.asked and hid not in self.knowledge:
            self.knowledge[hid] = intent
            self.answers.append(dict(t=round(t, 3), human=hid, intent=intent))
            return True
        return False

    def predict(self, hid):
        n = len(self.obs[hid])
        if self.p_cache.get(hid, (None,))[0] == n:
            return self.p_cache[hid][1]
        f = compute_features(self.obs[hid])
        if f is None:
            return None
        p = float(self.model.predict_proba(f.reshape(1, -1))[0, 1])
        if self.calibrator is not None:
            p = float(self.calibrator.transform(np.array([p]))[0])
        hist = self.p_hist.setdefault(hid, [])
        hist.append(p)
        del hist[:-self.smooth_n]
        p = float(np.mean(hist))
        self.p_cache[hid] = (n, p)
        return p

    def decide(self, t, robot_x):
        """Return (go, reason, ids_to_ask). Logs a decision row per engaged pedestrian."""
        go, reason, to_ask = True, 'no engaged pedestrian', []
        for hid, ob in self.obs.items():
            if len(ob) < 3:
                continue
            ox = float(np.mean([o[1] for o in ob[-3:]]))
            oy = float(np.mean([o[2] for o in ob[-3:]]))
            if hid not in self.cleared and (1.0 if oy >= 0 else -1.0) != self.side0[hid] and abs(oy) > self.clear_dist:
                self.cleared.add(hid)
            ahead = robot_x - ox
            if hid in self.cleared or not (-0.5 <= ahead <= self.engage_dist):
                continue

            p, action, why = None, 'go', ''
            known = self.knowledge.get(hid)
            at_deadline = ahead <= self.ask_dist
            waiting = hid in self.asked and known is None and t - self.asked[hid] < self.answer_timeout

            def ask():
                self.asked[hid] = t
                self.asks.append(dict(t=round(t, 3), human=hid))
                to_ask.append(hid)

            if self.policy == 'always_act':
                why = 'always act'
            elif self.policy == 'oracle':
                action, why = ('stop', 'oracle: crossing') if self.true_intent == 'crossing' else ('go', 'oracle: not crossing')
            elif self.policy == 'always_cautious':
                if at_deadline and abs(oy) < self.caution_dist:
                    action, why = 'stop', 'cautious: pedestrian near the line'
            elif known is not None:
                action, why = ('stop', 'answer: crossing') if known == 'crossing' else ('go', 'answer: not crossing')
            elif self.policy == 'always_ask':
                if hid not in self.asked and not at_deadline:
                    why = 'observing until the deadline'
                elif hid not in self.asked:
                    ask()
                    action, why = 'stop', 'asked'
                elif waiting:
                    action, why = 'stop', 'waiting for answer'
                else:
                    action, why = 'stop', 'no answer: cautious'
            else:  # threshold
                p = self.predict(hid)
                if p is None:
                    action, why = 'stop', 'not enough history'
                elif p >= self.p_high:
                    action, why = 'stop', 'confident: crossing'
                elif p <= self.p_low:
                    action, why = 'go', 'confident: not crossing'
                elif hid not in self.asked and not at_deadline:
                    why = 'uncertain: observing until the deadline'
                elif hid not in self.asked:
                    ask()
                    action, why = 'stop', 'uncertain: asked'
                elif waiting:
                    action, why = 'stop', 'uncertain: waiting for answer'
                else:
                    action, why = 'stop', 'uncertain, no answer: cautious'
            self.decisions.append((round(t, 3), hid, '' if p is None else round(p, 4), action, why,
                                   round(ahead, 3), round(abs(oy), 3)))
            if action == 'stop':
                go, reason = False, why
        return go, reason, to_ask

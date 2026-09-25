"""
Intent features for the Ask-or-Act study.

Computes features from a pedestrian's recently observed positions, relative to
the robot's path (the line y = line_y). The same function is used to build the
offline dataset and, later, by the robot at run time, so both see identical
features.

Features (all in the frame of the robot's line):
  dist_to_line     |y - line_y|                                    (m)
  v_toward_line    velocity component toward the line               (m/s, >0 = approaching)
  v_along_line     |velocity component parallel to the line|        (m/s)
  speed            overall speed                                    (m/s)
  a_toward_line    change in v_toward_line over the window          (m/s^2)
  stopped_frac     fraction of the last STOP_WINDOW s spent nearly still
"""
import numpy as np

FEATURE_NAMES = ['dist_to_line', 'v_toward_line', 'v_along_line', 'speed', 'a_toward_line', 'stopped_frac']

VEL_WINDOW = 1.0      # s of history used for velocity and acceleration
STOP_WINDOW = 2.0     # s of history used for stopped_frac
STOP_SPEED = 0.2      # m/s, below this counts as stopped
SPEED_LAG = 0.5       # s, displacement interval used to estimate speed for stopped_frac


def _fit_velocity(t, p):
    """Least-squares slope of p over t."""
    if len(t) < 2 or t[-1] - t[0] < 1e-6:
        return 0.0
    return float(np.polyfit(t, p, 1)[0])


def compute_features(hist, line_y=0.0):
    """hist: sequence of (t, x, y), oldest first. Returns a feature vector, or None if too little history."""
    h = np.asarray(hist, dtype=float)
    if len(h) < 4 or h[-1, 0] - h[0, 0] < VEL_WINDOW - 1e-6:
        return None
    t_now = h[-1, 0]
    w = h[h[:, 0] >= t_now - VEL_WINDOW - 1e-9]
    t, x, y = w[:, 0], w[:, 1], w[:, 2]

    y_now = float(np.mean(y[-3:]))                       # light smoothing against tracking noise
    side = np.sign(y_now - line_y) or -1.0                # which side of the line the pedestrian is on
    vx, vy = _fit_velocity(t, x), _fit_velocity(t, y)
    v_toward = -side * vy
    half = t_now - VEL_WINDOW / 2
    first, second = w[t < half], w[t >= half]
    a_toward = -side * (_fit_velocity(second[:, 0], second[:, 2]) -
                        _fit_velocity(first[:, 0], first[:, 2])) / (VEL_WINDOW / 2)

    s = h[h[:, 0] >= t_now - STOP_WINDOW - 1e-9]
    stopped = []
    for i in range(len(s)):
        j = np.searchsorted(h[:, 0], s[i, 0] - SPEED_LAG)
        if s[i, 0] - h[j, 0] < SPEED_LAG * 0.5:
            continue
        v = np.hypot(s[i, 1] - h[j, 1], s[i, 2] - h[j, 2]) / (s[i, 0] - h[j, 0])
        stopped.append(v < STOP_SPEED)
    stopped_frac = float(np.mean(stopped)) if stopped else 0.0

    return np.array([abs(y_now - line_y), v_toward, abs(vx), float(np.hypot(vx, vy)), a_toward, stopped_frac])

"""
Deterministic pedestrian trajectories for the Ask-or-Act study.

A pedestrian follows a plan of steps from a start point:
  {"type": "walk", "to": [x, y], "stop": true}    walk to a point (stop there, or keep moving)
  {"type": "wait", "duration": 2.0}               stand still

Speed follows a trapezoidal profile (accelerate, cruise, decelerate) so motion
looks natural, and everything is a pure function of time since release, so the
same scenario always produces the same trajectory. Both the scenario generator
(for timing) and the pedestrian node (for motion) use this module.
"""
import bisect
import math

HUMAN_RADIUS = 0.3        # m, body radius of the simulated pedestrian
ACCEL = 0.8               # m/s^2, acceleration and deceleration
TURN_SPEED_FACTOR = 0.6   # fraction of walking speed kept through a non-stop waypoint
DT = 0.01                 # s, trajectory resolution


def build_trajectory(start, plan, speed, accel=ACCEL, dt=DT):
    """Return a list of samples (t, x, y, yaw, v), t = seconds since release."""
    x, y = start
    yaw = 0.0
    for step in plan:
        if step['type'] == 'walk':
            yaw = math.atan2(step['to'][1] - y, step['to'][0] - x)
            break
    t, v = 0.0, 0.0
    out = [(0.0, x, y, yaw, 0.0)]
    for step in plan:
        if step['type'] == 'wait':
            v = 0.0
            for _ in range(int(round(step['duration'] / dt))):
                t += dt
                out.append((t, x, y, yaw, 0.0))
            continue
        tx, ty = step['to']
        length = math.hypot(tx - x, ty - y)
        if length < 1e-6:
            continue
        ux, uy = (tx - x) / length, (ty - y) / length
        yaw = math.atan2(uy, ux)
        stop = step.get('stop', True)
        vf = 0.0 if stop else speed * TURN_SPEED_FACTOR
        x0, y0, s = x, y, 0.0
        while length - s > 1e-4:
            rem = length - s
            brake = (v * v - vf * vf) / (2.0 * accel) if v > vf else 0.0
            if rem <= brake:
                v = max(vf, v - accel * dt)
            else:
                v = min(speed, v + accel * dt)
            v = max(v, 0.05)                 # never stall just short of a waypoint
            s += min(v * dt, rem)
            t += dt
            x, y = x0 + ux * s, y0 + uy * s
            out.append((t, x, y, yaw, v))
        v = 0.0 if stop else v
    return out


def pose_at(traj, times, t):
    """Pose (x, y, yaw, v) at time t since release; holds the start/end pose outside the plan."""
    if t <= 0.0:
        _, x, y, yaw, _ = traj[0]
        return x, y, yaw, 0.0
    i = bisect.bisect_right(times, t) - 1
    if i >= len(traj) - 1:
        _, x, y, yaw, _ = traj[-1]
        return x, y, yaw, 0.0
    _, x, y, yaw, v = traj[i]
    return x, y, yaw, v


def first_line_crossing(traj, line_y=0.0):
    """(t, x) when the pedestrian first crosses y = line_y, or None."""
    below = traj[0][2] < line_y
    for t, x, y, _, _ in traj[1:]:
        if (y < line_y) != below:
            return t, x
    return None

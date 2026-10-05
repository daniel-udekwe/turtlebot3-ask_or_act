#!/usr/bin/env python3
"""
Onboard computation benchmark for the Ask-or-Act decision pipeline.

Runs the exact decision code used in the experiments (intent_features.py,
the trained MLP, the decision-aware calibrator, policy.py) on realistic,
noisy pedestrian observations generated from the scenario families, and
measures:

  components   time per prediction step for feature extraction, the MLP,
               calibration, and in total (median, p95, p99, max)
  decision     time per call of the full decision rule at 20 Hz
  real time    the decision rule running at its real 20 Hz rate for a fixed
               duration: deadline misses (> 50 ms) and CPU use
  footprint    model file size, model loading time, peak memory

Results are printed and saved as JSON together with the hardware and
software versions. Run it on the robot's computer and on a desktop.

Files needed alongside this script: policy.py, intent_features.py,
calibration.py, ped_traj.py, generate_scenarios.py, and the trained model
(predictor/intent_model.pkl). The model must be loaded with the same
scikit-learn version it was trained with.

Usage:
  python3 benchmark_onboard.py
  python3 benchmark_onboard.py --model-file intent_model.pkl --realtime-seconds 120
"""
import argparse
import json
import os
import platform
import random
import resource
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_scenarios import CONDITIONS, ENVS, FAMILIES, ROBOT, build, eval_seeds   # noqa: E402
from intent_features import compute_features                                        # noqa: E402
from ped_traj import build_trajectory, pose_at                                      # noqa: E402
from policy import AskOrActPolicy, load_predictor                                   # noqa: E402

OBS_DT, CTRL_DT = 0.1, 0.05        # 10 Hz tracking, 20 Hz decisions (as in the experiments)
HISTORY = 40


# ------------------------------------------------------------------ system info
def read(path):
    try:
        with open(path) as f:
            return f.read().strip('\x00\n ')
    except OSError:
        return None


def device_info():
    fields = {}
    for line in (read('/proc/cpuinfo') or '').splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            fields.setdefault(k.strip().lower(), v.strip())
    # PCs report 'model name'; Raspberry Pi kernels report 'Model' (or 'Hardware')
    cpu = fields.get('model name') or fields.get('hardware') or fields.get('model')
    if platform.machine() in ('aarch64', 'armv7l') and not fields.get('model name'):
        cpu = read('/proc/device-tree/model') or cpu
    mem_kb = next((int(l.split()[1]) for l in (read('/proc/meminfo') or '').splitlines()
                   if l.startswith('MemTotal')), None)
    import sklearn
    return dict(host=socket.gethostname(), board=read('/proc/device-tree/model'), cpu=cpu,
                cores=os.cpu_count(), machine=platform.machine(), os=platform.platform(),
                memory_gb=round(mem_kb / 1e6, 2) if mem_kb else None, python=platform.python_version(),
                numpy=np.__version__, sklearn=sklearn.__version__)


# ------------------------------------------------------------------ observations
def episodes(condition, seeds, rng):
    """Noisy 10 Hz observation streams (t, x, y) and robot positions for every family."""
    noise = CONDITIONS[condition]['noise']
    out = []
    for env in ENVS:
        for family, intent in FAMILIES:
            for seed in eval_seeds(condition, 7000, seeds):
                lab = build(env, family, intent, seed, condition)
                traj = build_trajectory(lab['start'], lab['plan'], lab['human_speed'])
                ts = [s[0] for s in traj]
                obs, robot = [], []
                t = -2.0
                while t <= 12.0:
                    x, y, _, _ = pose_at(traj, ts, t - lab['release_delay'])
                    obs.append((t, x + rng.gauss(0, noise), y + rng.gauss(0, noise)))
                    robot.append(ROBOT['start'][0] - ROBOT['speed'] * max(0.0, t - lab['robot_delay']))
                    t += OBS_DT
                out.append((obs, robot))
    return out


def summary_ms(samples):
    a = np.asarray(samples) * 1e3
    return dict(n=int(a.size), median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), max=float(a.max()), mean=float(a.mean()))


# ------------------------------------------------------------------ benchmarks
def bench_components(model, cal, eps, warmup=200):
    t_feat, t_mlp, t_cal, t_total = [], [], [], []
    n = 0
    for obs, _ in eps:
        for k in range(len(obs)):
            hist = obs[max(0, k - HISTORY + 1):k + 1]
            t0 = time.perf_counter()
            f = compute_features(hist)
            t1 = time.perf_counter()
            if f is None:
                continue
            p = model.predict_proba(f.reshape(1, -1))[0, 1]
            t2 = time.perf_counter()
            cal.transform(np.array([p]))
            t3 = time.perf_counter()
            n += 1
            if n <= warmup:
                continue
            t_feat.append(t1 - t0)
            t_mlp.append(t2 - t1)
            t_cal.append(t3 - t2)
            t_total.append(t3 - t0)
    return dict(features=summary_ms(t_feat), mlp=summary_ms(t_mlp), calibration=summary_ms(t_cal),
                total=summary_ms(t_total))


def bench_decision(model, cal, eps):
    """Full decision rule (policy.decide) at 20 Hz, observations added at 10 Hz."""
    times = []
    for obs, robot in eps:
        pol = AskOrActPolicy('threshold', model, cal)
        for k in range(len(obs)):
            pol.add_observation(1, *obs[k])
            for _ in range(2):                      # two 20 Hz decisions per 10 Hz observation
                t0 = time.perf_counter()
                pol.decide(obs[k][0], robot[k])
                times.append(time.perf_counter() - t0)
    return summary_ms(times)


def bench_realtime(model, cal, eps, seconds):
    """Run decisions at the real 20 Hz rate, episode after episode, for `seconds`;
    count deadline misses and measure CPU use."""
    compute, misses, ticks = [], 0, 0
    cpu0, wall0 = time.process_time(), time.perf_counter()
    next_tick = wall0
    e = 0
    while time.perf_counter() - wall0 < seconds:
        obs, robot = eps[e % len(eps)]
        e += 1
        pol = AskOrActPolicy('threshold', model, cal)        # a fresh encounter
        for k in range(2 * len(obs)):                        # 20 Hz ticks, observations at 10 Hz
            if time.perf_counter() - wall0 >= seconds:
                break
            t0 = time.perf_counter()
            if k % 2 == 0:
                pol.add_observation(1, *obs[k // 2])
            pol.decide(obs[k // 2][0], robot[k // 2])
            dt = time.perf_counter() - t0
            compute.append(dt)
            misses += dt > CTRL_DT
            ticks += 1
            next_tick += CTRL_DT
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    return dict(seconds=round(wall, 1), ticks=ticks, episodes=e, rate_hz=round(ticks / wall, 2),
                deadline_ms=CTRL_DT * 1e3, deadline_misses=int(misses),
                cpu_percent_of_one_core=round(100 * cpu / wall, 2), compute=summary_ms(compute))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(here, 'predictor', 'intent_model.pkl')
    if not os.path.exists(default_model):
        default_model = os.path.join(here, 'intent_model.pkl')
    ap.add_argument('--model-file', default=default_model)
    ap.add_argument('--model', default='mlp')
    ap.add_argument('--conditions', nargs='+', default=['nominal', 'noisy'])
    ap.add_argument('--seeds', type=int, default=10, help='scenarios per family and condition')
    ap.add_argument('--realtime-seconds', type=float, default=60.0)
    ap.add_argument('--out', default=None, help='JSON output (default: benchmark_<host>.json)')
    args = ap.parse_args()

    info = device_info()
    print('Device:', ', '.join(f'{k}={v}' for k, v in info.items() if v))
    t0 = time.perf_counter()
    model, _ = load_predictor(args.model_file, args.model, 'raw')
    load_s = time.perf_counter() - t0
    results = dict(device=info, model_file=os.path.basename(args.model_file),
                   model_file_kb=round(os.path.getsize(args.model_file) / 1024, 1),
                   model_load_s=round(load_s, 3), conditions={})

    rng = random.Random(0)
    for cond in args.conditions:
        _, cal = load_predictor(args.model_file, args.model, 'isotonic_dw', cond)
        eps = episodes(cond, args.seeds, rng)
        print(f'\n[{cond}] {len(eps)} episodes: timing components ...')
        comp = bench_components(model, cal, eps)
        print(f'[{cond}] timing the full decision rule ...')
        dec = bench_decision(model, cal, eps)
        results['conditions'][cond] = dict(components=comp, decision=dec)

    cond = args.conditions[0]
    _, cal = load_predictor(args.model_file, args.model, 'isotonic_dw', cond)
    print(f'\nReal-time test: decisions at 20 Hz for {args.realtime_seconds:.0f} s ...')
    results['realtime'] = bench_realtime(model, cal, episodes(cond, args.seeds, rng), args.realtime_seconds)
    results['peak_memory_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

    print(f"\nModel: {results['model_file_kb']} kB, loaded in {results['model_load_s']} s; "
          f"peak memory {results['peak_memory_mb']} MB")
    print(f"{'condition':<9}{'step':<14}{'median':>9}{'p95':>9}{'p99':>9}{'max':>9}   (ms)")
    for c, r in results['conditions'].items():
        rows = list(r['components'].items()) + [('decision', r['decision'])]
        for name, s in rows:
            print(f"{c:<9}{name:<14}{s['median']:>9.3f}{s['p95']:>9.3f}{s['p99']:>9.3f}{s['max']:>9.3f}")
    rt = results['realtime']
    print(f"\nReal time: {rt['ticks']} decisions in {rt['seconds']} s ({rt['rate_hz']} Hz); "
          f"deadline misses (> {rt['deadline_ms']:.0f} ms): {rt['deadline_misses']}; "
          f"CPU {rt['cpu_percent_of_one_core']}% of one core; "
          f"compute median {rt['compute']['median']:.3f} ms, max {rt['compute']['max']:.3f} ms")

    out = args.out or os.path.join(here, f"benchmark_{info['host']}.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()

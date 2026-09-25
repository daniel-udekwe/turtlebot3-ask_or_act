#!/usr/bin/env python3
"""
Episode runner for the Ask-or-Act study (scripted pedestrians).

For each scenario in labels.json it starts a fresh headless Gazebo with the
environment's world, spawns the robot, starts scripted_pedestrians.py, runs
line_controller.py with the scenario's start offsets, appends a row to
runs/results.csv, and shuts everything down before the next scenario.

Policies (see policy.py): always_act (default), always_cautious, always_ask, oracle,
threshold (with --calibration raw|isotonic|temperature and --calib-condition stale|recal).

Usage:
  python3 run_episodes.py                           # all scenarios, always_act
  python3 run_episodes.py --policy threshold --calibration isotonic --calib-condition recal
  python3 run_episodes.py --only 'cafe_hs_*'        # subset (shell-style pattern)
  python3 run_episodes.py --resume                  # skip episodes already in results.csv
  python3 run_episodes.py --only cafe_cc_x_00 --gui # watch one episode in the Gazebo window
"""
import argparse
import csv
import fnmatch
import json
import os
import signal
import subprocess
import time

from generate_scenarios import CONDITIONS

HOME = os.path.expanduser('~')
BASE = os.path.join(HOME, 'ask_or_act')
SCEN = os.path.join(BASE, 'scenarios')
WORLDS = os.path.join(BASE, 'worlds')
CONTROLLER = os.path.join(BASE, 'line_controller.py')
PEDESTRIANS = os.path.join(BASE, 'scripted_pedestrians.py')
ROBOT_SDF = os.path.join(HOME, 'burger_novis.sdf')
GZ_PARAMS = os.path.join(BASE, 'gazebo_params.yaml')

ROBOT_START = (2.9, 0.1)      # must match line_controller.py
ROBOT_SPEED = 0.18
COLLISION_DIST = 0.4          # pedestrian radius 0.3 m + Burger radius ~0.1 m

STATE_PLUGIN = ('<plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">'
                '<ros><namespace>/</namespace></ros></plugin>')


def pkg_share(pkg):
    prefix = subprocess.run(['ros2', 'pkg', 'prefix', pkg], capture_output=True, text=True, check=True)
    return os.path.join(prefix.stdout.strip(), 'share', pkg)


def gazebo_env():
    wrapper, tb3 = pkg_share('hunav_gazebo_wrapper'), pkg_share('turtlebot3_gazebo')
    env = dict(os.environ)
    models = [wrapper, os.path.join(wrapper, 'models'), os.path.join(tb3, 'models'), '/usr/share/gazebo-11/models']
    resources = ['/usr/share/gazebo-11', wrapper, os.path.join(wrapper, 'models'), os.path.join(wrapper, 'worlds')]
    env['GAZEBO_MODEL_PATH'] = ':'.join(models + [env.get('GAZEBO_MODEL_PATH', '')])
    env['GAZEBO_RESOURCE_PATH'] = ':'.join(resources + [env.get('GAZEBO_RESOURCE_PATH', '')])
    return env, wrapper


def ensure_setup(env_name, wrapper):
    """Create the world with the entity-state plugin, and the Gazebo parameter file."""
    os.makedirs(WORLDS, exist_ok=True)
    dst = os.path.join(WORLDS, f'{env_name}_scripted.world')
    if not os.path.exists(dst):
        src = open(os.path.join(wrapper, 'worlds', f'{env_name}.world')).read()
        i = src.index('<world')
        j = src.index('>', i) + 1
        open(dst, 'w').write(src[:j] + '\n    ' + STATE_PLUGIN + src[j:])
        print(f'Created {dst}')
    if not os.path.exists(GZ_PARAMS):
        open(GZ_PARAMS, 'w').write('gazebo:\n  ros__parameters:\n    use_sim_time: true\n'
                                   '    publish_rate: 100.0\n')
    return dst


def quiet(cmd):
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def kill_sim():
    quiet(['killall', '-9', 'gzserver', 'gzclient'])
    quiet(['pkill', '-9', '-f', 'scripted_pedestrians.py|line_controller.py'])


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def policy_params(args, label):
    """ROS parameters for the controller's decision policy."""
    cond = label.get('condition', 'nominal')
    calib_cond = 'nominal' if args.calib_condition == 'stale' else cond
    return ['-p', f'policy:={args.policy}', '-p', f'model_name:={args.model}',
            '-p', f'calibration:={args.calibration}', '-p', f'calib_condition:={calib_cond}',
            '-p', f'p_low:={args.p_low}', '-p', f'p_high:={args.p_high}',
            '-p', f'obs_noise:={CONDITIONS[cond]["noise"]}', '-p', f"noise_seed:={label['id']}",
            '-p', f"true_intent:={label['intent']}"]


def run_episode(label, world, env, ep_dir, gui, args):
    """Run one episode; return the controller's result dict, or None on failure."""
    os.makedirs(ep_dir, exist_ok=True)
    res_path = os.path.join(ep_dir, 'result.json')
    if os.path.exists(res_path):
        os.remove(res_path)
    logs = {n: open(os.path.join(ep_dir, f'{n}.log'), 'w') for n in ('gazebo', 'spawn', 'pedestrians', 'controller')}
    procs = []
    kill_sim()
    time.sleep(1.0)
    try:
        gz = subprocess.Popen(['gzserver', world, '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so',
                               '--ros-args', '--params-file', GZ_PARAMS],
                              env=env, stdout=logs['gazebo'], stderr=subprocess.STDOUT, start_new_session=True)
        procs.append(gz)
        if gui:
            procs.append(subprocess.Popen(['gzclient'], env=env, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, start_new_session=True))
        subprocess.run(['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', '-entity', 'burger', '-file', ROBOT_SDF,
                        '-x', str(ROBOT_START[0]), '-y', str(ROBOT_START[1]), '-z', '0.05', '-Y', '3.14159',
                        '-timeout', '120'],
                       env=env, stdout=logs['spawn'], stderr=subprocess.STDOUT, timeout=180, check=True)
        peds = subprocess.Popen(['python3', PEDESTRIANS, '--ros-args', '-p', 'use_sim_time:=true',
                                 '-p', f"scenario_id:={label['id']}", '-p', f'labels_file:={args.labels}',
                                 '-p', f'response_delay:={args.response_delay}',
                                 '-p', f'response_prob:={args.response_prob}',
                                 '-p', f'answer_accuracy:={args.answer_accuracy}'],
                                env=env, stdout=logs['pedestrians'], stderr=subprocess.STDOUT, start_new_session=True)
        procs.append(peds)
        time.sleep(4.0)                      # let the pedestrian spawn before the episode starts
        subprocess.run(['python3', CONTROLLER, '--ros-args', '-p', 'use_sim_time:=true',
                        '-p', f'speed:={ROBOT_SPEED:.3f}',
                        '-p', f"human_release_delay:={label['release_delay']:.3f}",
                        '-p', f"robot_start_delay:={label['robot_delay']:.3f}",
                        '-p', 'max_duration:=60.0',
                        '-p', f'result_file:={res_path}',
                        '-p', f"traj_file:={os.path.join(ep_dir, 'trajectory.csv')}",
                        '-p', f"decisions_file:={os.path.join(ep_dir, 'decisions.csv')}"] + policy_params(args, label),
                       env=env, stdout=logs['controller'], stderr=subprocess.STDOUT, timeout=400)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f'   episode failed: {e}')
    finally:
        for p in reversed(procs):
            stop(p)
        for f in logs.values():
            f.close()
        kill_sim()
    if not os.path.exists(res_path):
        return None
    return json.load(open(res_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', default='*', help="scenario id pattern, e.g. 'cafe_hs_*'")
    ap.add_argument('--labels', default=os.path.join(SCEN, 'labels.json'), help='scenario labels file')
    ap.add_argument('--out', default=os.path.join(BASE, 'runs'))
    ap.add_argument('--policy', default='always_act',
                    choices=['always_act', 'always_cautious', 'always_ask', 'threshold', 'oracle'])
    ap.add_argument('--model', default='mlp')
    ap.add_argument('--calibration', default='isotonic', choices=['raw', 'isotonic', 'isotonic_dw', 'temperature'])
    ap.add_argument('--calib-condition', default='recal', choices=['stale', 'recal'],
                    help='stale: calibrator fitted on nominal data; recal: fitted on the scenario condition')
    ap.add_argument('--p-low', type=float, default=0.2)
    ap.add_argument('--p-high', type=float, default=0.8)
    ap.add_argument('--response-delay', type=float, default=1.5)
    ap.add_argument('--response-prob', type=float, default=1.0)
    ap.add_argument('--answer-accuracy', type=float, default=1.0)
    ap.add_argument('--method', default='', help='label for this batch in results.csv (default: derived)')
    ap.add_argument('--resume', action='store_true', help='skip episodes already in results.csv')
    ap.add_argument('--gui', action='store_true', help='also open the Gazebo window (slow)')
    args = ap.parse_args()

    if not args.method:
        args.method = args.policy if args.policy != 'threshold' else (
            'threshold:raw' if args.calibration == 'raw' else f'threshold:{args.calibration}@{args.calib_condition}')
    args.labels = os.path.abspath(os.path.expanduser(args.labels))
    labels = json.load(open(args.labels))
    labels = [l for l in labels if fnmatch.fnmatch(l['id'], args.only)]
    env, wrapper = gazebo_env()
    os.makedirs(args.out, exist_ok=True)
    res_csv = os.path.join(args.out, 'results.csv')
    cols = ['id', 'method', 'condition', 'env', 'family', 'intent', 'ambiguous', 'robot_dist_at_commit',
            'release_delay', 'robot_delay', 'outcome', 'duration', 'min_dist', 'collision',
            'n_asks', 'stopped_time', 'cross_t', 'cross_x', 'robot_dist_at_cross']
    done = set()
    if args.resume and os.path.exists(res_csv):
        done = {(r['id'], r['method']) for r in csv.DictReader(open(res_csv))}
    new_file = not os.path.exists(res_csv)
    print(f'{len(labels)} scenarios selected')

    with open(res_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        for n, lab in enumerate(labels, 1):
            if (lab['id'], args.method) in done:
                continue
            world = ensure_setup(lab['env'], wrapper)
            print(f"[{n}/{len(labels)}] {lab['id']}: release {lab['release_delay']:.2f} s, "
                  f"robot delay {lab['robot_delay']:.2f} s")
            res = run_episode(lab, world, env, os.path.join(args.out, args.method.replace(':', '_'), lab['id']),
                              args.gui, args)
            if res is None:
                print('   no result recorded; see the logs in', os.path.join(args.out, args.method, lab['id']))
                continue
            c = res['crossings'][0] if res['crossings'] else None
            row = dict(id=lab['id'], method=args.method, condition=lab.get('condition', 'nominal'),
                       env=lab['env'], family=lab['family'],
                       intent=lab['intent'], ambiguous=lab['ambiguous'],
                       robot_dist_at_commit=lab['robot_dist_at_commit'],
                       release_delay=lab['release_delay'], robot_delay=lab['robot_delay'],
                       outcome=res['outcome'], duration=res['duration'], min_dist=res['min_dist'],
                       collision=res['min_dist'] is not None and res['min_dist'] < COLLISION_DIST,
                       n_asks=res.get('n_asks', 0), stopped_time=res.get('stopped_time', 0.0),
                       cross_t=c['t'] if c else '', cross_x=c['x'] if c else '',
                       robot_dist_at_cross=round(c['robot_x'] - c['x'], 3) if c else '')
            w.writerow(row)
            f.flush()
            print(f"   {res['outcome']}, min distance {res['min_dist']} m, collision={row['collision']}, "
                  f"asks {row['n_asks']}, stopped {row['stopped_time']} s")
    print(f'Results in {res_csv}')


if __name__ == '__main__':
    main()

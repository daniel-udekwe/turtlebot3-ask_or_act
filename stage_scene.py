#!/usr/bin/env python3
"""
Stage a 3D scene in Gazebo for paper figures (static, paused; take a screenshot).

Modes:
  routes     every scenario family's route as a coloured ribbon on the floor
             (solid: crosses, dashed: does not; colours match fig_scenarios),
             the robot's path in black, the TurtleBot3 at its start, and
             pedestrian figures at the start and where pedestrians stop
  snapshots  one scenario as a time-lapse: the pedestrian at several moments
             (faint to solid) and the robot at a chosen moment

Pedestrians are shown as animated human figures (Gazebo actors, the standard
walk.dae figure) by default, or as simple cylinders with --figures cylinders.
Human figures walk the last half metre into place with the simulation running,
then the simulation is paused, so each one freezes mid-stride.

Leaves Gazebo open with the window. Take the picture with the camera icon in the
Gazebo toolbar (saved in ~/.gazebo/pictures), then press Ctrl+C here.

Usage:
  python3 stage_scene.py routes
  python3 stage_scene.py snapshots --scenario cafe_hs_x_7000 --times 0 2 3.5 5 6.5 --robot-time 5
  python3 stage_scene.py routes --camera 6 -6 5 --target 1 -0.5 0
  python3 stage_scene.py snapshots --scenario cafe_hs_x_7000 --times 0 2 3.5 5 6.5 --robot-time 5 \
      --trajectory ~/ask_or_act/runs/validation/always_ask/cafe_hs_x_7000/trajectory.csv
      (positions taken from a recorded Gazebo episode, e.g. the robot stopped and waiting)
"""
import argparse
import csv
import math
import os
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_scenarios import FAMILIES, ROBOT, build, eval_seeds   # noqa: E402
from ped_traj import HUMAN_RADIUS, build_trajectory, pose_at        # noqa: E402
from run_episodes import ROBOT_SDF, WORLDS, gazebo_env              # noqa: E402

# Okabe-Ito colours, matching make_figures.py (RGB, 0-1)
HEX = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7']
FAM_ORDER = list(dict.fromkeys(f for f, _ in FAMILIES))
rgb = lambda h: tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
FAM_RGB = {f: rgb(HEX[i]) for i, f in enumerate(FAM_ORDER)}


def material(r, g, b, transparency=0.0):
    return (f'<material><ambient>{r:.3f} {g:.3f} {b:.3f} 1</ambient><diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse>'
            f'<emissive>{0.35 * r:.3f} {0.35 * g:.3f} {0.35 * b:.3f} 1</emissive></material>'
            f'<transparency>{transparency:.2f}</transparency>')


def box_xml(name, length, width, height, color, transparency=0.0):
    """Static, visual-only box (no collision), so it never affects the simulation."""
    return (f'<?xml version="1.0"?><sdf version="1.6"><model name="{name}"><static>true</static><link name="l">'
            f'<visual name="v"><geometry><box><size>{length:.3f} {width:.3f} {height:.3f}</size></box></geometry>'
            f'{material(*color, transparency)}</visual></link></model></sdf>')


def person_xml(name, color, transparency=0.0):
    r, g, b = color
    return (f'<?xml version="1.0"?><sdf version="1.6"><model name="{name}"><static>true</static><link name="body">'
            f'<visual name="torso"><pose>0 0 0.78 0 0 0</pose><geometry><cylinder><radius>{HUMAN_RADIUS}</radius>'
            f'<length>1.56</length></cylinder></geometry>{material(r, g, b, transparency)}</visual>'
            f'<visual name="head"><pose>0 0 1.68 0 0 0</pose><geometry><sphere><radius>0.12</radius></sphere>'
            f'</geometry>{material(0.9, 0.75, 0.6, transparency)}</visual></link></model></sdf>')


FLOOR_Z = 0.04        # ribbon height above z = 0; the cafe floor surface sits above the ground plane


def actor_xml(name, x, y, yaw, z=0.0, yaw_offset=0.0, approach=0.5, skin='walk.dae', scale=1.0):
    """Animated human (Gazebo actor) that walks `approach` m into (x, y), facing yaw, then stands still.
    With interpolate_x the animation only advances while moving, so it freezes mid-stride."""
    x0, y0 = x - approach * math.cos(yaw), y - approach * math.sin(yaw)
    a = yaw + yaw_offset
    wp = lambda t, px, py: f'<waypoint><time>{t}</time><pose>{px:.3f} {py:.3f} {z:.3f} 0 0 {a:.4f}</pose></waypoint>'
    return (f'<actor name="{name}"><pose>{x0:.3f} {y0:.3f} {z:.3f} 0 0 {a:.4f}</pose>'
            f'<skin><filename>{skin}</filename><scale>{scale:.3f}</scale></skin>'
            f'<animation name="walking"><filename>{skin}</filename><scale>{scale:.3f}</scale>'
            f'<interpolate_x>true</interpolate_x></animation>'
            f'<script><loop>false</loop><auto_start>true</auto_start><trajectory id="0" type="walking">'
            f'{wp(0.0, x0, y0)}{wp(0.6, x, y)}{wp(100000.0, x, y)}</trajectory></script></actor>')


def ribbon(points, name, color, dashed=False, width=0.06, dash=0.18, gap=0.12):
    """Floor ribbon along a polyline, as (name, xml, x, y, z, yaw) items."""
    items = []
    k = 0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-6:
            continue
        yaw = math.atan2(y1 - y0, x1 - x0)
        pieces = [(0.0, seg)] if not dashed else [(s, min(s + dash, seg)) for s in
                                                   [i * (dash + gap) for i in range(int(seg / (dash + gap)) + 1)]
                                                   if s < seg]
        for a, b in pieces:
            mid = (a + b) / 2
            items.append((f'{name}_{k}', box_xml(f'{name}_{k}', b - a, width, 0.01, color),
                          x0 + mid * math.cos(yaw), y0 + mid * math.sin(yaw), FLOOR_Z, yaw))
            k += 1
    return items


def routes_scene(offset_step=0.12):
    items, people = [], []
    items += ribbon([ROBOT['start'], ROBOT['goal']], 'robot_path', (0.0, 0.0, 0.0), width=0.08)
    for fam, intent in FAMILIES:
        lab = build('cafe', fam, intent, eval_seeds('nominal', 7000, 1)[0], 'nominal')
        dx = offset_step * (FAM_ORDER.index(fam) - (len(FAM_ORDER) - 1) / 2) - lab['route_shift']
        pts = [(p[0] + dx, p[1]) for p in [lab['start']] + lab['waypoints']]
        items += ribbon(pts, f'route_{fam}_{intent}', FAM_RGB[fam], dashed=(intent != 'crossing'))
        if fam in ('hesitant', 'wait_far') and intent == 'crossing' or fam == 'wait_far':
            stop = pts[1]
            people.append((f'stop_{fam}', person_xml(f'stop_{fam}', FAM_RGB[fam]), stop[0], stop[1], math.pi / 2))
    start = build('cafe', 'clear_cross', 'crossing', eval_seeds('nominal', 7000, 1)[0], 'nominal')['start']
    people.append(('start_person', person_xml('start_person', (0.55, 0.55, 0.6)), start[0], start[1] - 0.45, math.pi / 2))
    return items, people, (ROBOT['start'][0], ROBOT['start'][1], math.pi)


def from_trajectory(path, times, robot_time):
    """Pedestrian and robot poses at the given times from a recorded episode (trajectory.csv)."""
    rows = [dict((k, float(v)) for k, v in r.items()) for r in csv.DictReader(open(os.path.expanduser(path)))]
    at = lambda t: min(rows, key=lambda r: abs(r['t'] - t))
    peds = []
    for t in times:
        r, r2 = at(t), at(t + 0.3)
        yaw = math.atan2(r2['human_y'] - r['human_y'], r2['human_x'] - r['human_x']) \
            if math.hypot(r2['human_y'] - r['human_y'], r2['human_x'] - r['human_x']) > 0.02 else math.pi / 2
        peds.append((r['human_x'], r['human_y'], yaw))
    rr = at(robot_time)
    return peds, (rr['robot_x'], rr['robot_y'], math.pi)


def snapshots_scene(scenario, times, robot_time, trajectory=None):
    parts = scenario.split('_')
    family = {'cc': 'clear_cross', 'cn': 'clear_noncross', 'hs': 'hesitant', 'lt': 'late_turn',
              'sc': 'slow_cross', 'wf': 'wait_far'}[parts[1]]
    intent = 'crossing' if parts[2] == 'x' else 'not_crossing'
    cond = parts[4] if len(parts) > 4 else 'nominal'
    lab = build(parts[0], family, intent, int(parts[3]), cond)
    traj = build_trajectory(lab['start'], lab['plan'], lab['human_speed'])
    ts = [s[0] for s in traj]
    color = FAM_RGB[family]
    items = ribbon([ROBOT['start'], ROBOT['goal']], 'robot_path', (0.0, 0.0, 0.0), width=0.08)
    items += ribbon([lab['start']] + lab['waypoints'], 'route', color, dashed=(intent != 'crossing'))
    if trajectory:
        poses, (rx, ry, _) = from_trajectory(trajectory, times, robot_time)
        source = f'recorded episode {trajectory}'
    else:
        poses = [pose_at(traj, ts, t - lab['release_delay'])[:3] for t in times]
        # robot where it would be had it driven without stopping (illustration only)
        rx = ROBOT['start'][0] - ROBOT['speed'] * max(0.0, robot_time - lab['robot_delay'])
        frac = (ROBOT['start'][0] - rx) / (ROBOT['start'][0] - ROBOT['goal'][0])
        ry = ROBOT['start'][1] + frac * (ROBOT['goal'][1] - ROBOT['start'][1])
        source = 'planned trajectory, robot never stopping'
    people = []
    for i, (x, y, yaw) in enumerate(poses):            # times since episode start; faint to solid
        transp = 0.55 * (1 - i / max(1, len(poses) - 1))   # earliest faintest, but still visible
        people.append((f'ghost_{i}', person_xml(f'ghost_{i}', color, transp), x, y, yaw))
    print(f"{scenario}: {family} / {intent}; positions from {source}; robot shown at t = {robot_time} s "
          f"(x = {rx:.2f})")
    return items, people, (rx, ry, math.pi)


def stage_world(camera, target, extra=''):
    """Copy the scripted world with the GUI camera placed at `camera`, looking at `target`,
    plus any extra SDF (e.g. actors, which must be part of the world file)."""
    src = os.path.join(WORLDS, 'cafe_scripted.world')
    if not os.path.exists(src):
        raise SystemExit('Run run_episodes.py once first (it creates worlds/cafe_scripted.world).')
    cx, cy, cz = camera
    tx, ty, tz = target
    yaw = math.atan2(ty - cy, tx - cx)
    pitch = math.atan2(cz - tz, math.hypot(tx - cx, ty - cy))
    gui = (f'<gui><camera name="user_camera"><pose>{cx} {cy} {cz} 0 {pitch:.4f} {yaw:.4f}</pose>'
           f'</camera></gui>')
    s = open(src).read()
    j = s.index('>', s.index('<world')) + 1
    s = s[:j] + '\n    ' + gui + s[j:]
    k = s.rindex('</world>')
    s = s[:k] + extra + '\n  ' + s[k:]
    dst = os.path.join(WORLDS, 'cafe_stage.world')
    open(dst, 'w').write(s)
    return dst


class Spawner(Node):
    def __init__(self):
        super().__init__('scene_spawner')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')

    def spawn(self, name, xml, x, y, z, yaw):
        req = SpawnEntity.Request()
        req.name, req.xml = name, xml
        req.initial_pose.position.x, req.initial_pose.position.y, req.initial_pose.position.z = x, y, z
        req.initial_pose.orientation.z, req.initial_pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=30.0)
        return fut.result() is not None and fut.result().success


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['routes', 'snapshots'])
    ap.add_argument('--scenario', default='cafe_hs_x_7000', help='for snapshots')
    ap.add_argument('--times', nargs='+', type=float, default=[0.0, 2.0, 3.5, 5.0, 6.5],
                    help='snapshot times (s since episode start)')
    ap.add_argument('--robot-time', type=float, default=5.0, help='where to place the robot (snapshots)')
    ap.add_argument('--trajectory', default=None, help='trajectory.csv of a recorded episode (snapshots)')
    ap.add_argument('--floor-z', type=float, default=FLOOR_Z, help='height of the floor ribbons (m)')
    ap.add_argument('--figures', choices=['humans', 'cylinders'], default='humans')
    ap.add_argument('--human-scale', type=float, default=1.0, help='size of the human figures (e.g. 0.9 = 10%% smaller)')
    ap.add_argument('--no-ribbons', action='store_true', help='leave out the robot path and route ribbons')
    ap.add_argument('--actor-z', type=float, default=0.0, help='raise/lower the human figures (m) if they float or sink')
    ap.add_argument('--actor-yaw-offset', type=float, default=0.0,
                    help='rotate the human figures (rad) if they face sideways')
    ap.add_argument('--camera', nargs=3, type=float, default=[6.0, -6.0, 5.0], metavar=('X', 'Y', 'Z'))
    ap.add_argument('--target', nargs=3, type=float, default=[1.0, -0.5, 0.0], metavar=('X', 'Y', 'Z'))
    args = ap.parse_args()
    globals()['FLOOR_Z'] = args.floor_z

    if args.mode == 'routes':
        items, people, robot = routes_scene()
    else:
        items, people, robot = snapshots_scene(args.scenario, args.times, args.robot_time, args.trajectory)

    if args.no_ribbons:
        items = []
    humans = args.figures == 'humans'
    actors = ''
    if humans:
        actors = ''.join('\n    ' + actor_xml(name, x, y, yaw, args.actor_z, args.actor_yaw_offset,
                                               scale=args.human_scale)
                         for name, _, x, y, yaw in people)
        people = []
    env, _ = gazebo_env()
    world = stage_world(args.camera, args.target, actors)
    subprocess.run(['killall', '-9', 'gzserver', 'gzclient'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    server = subprocess.Popen(['gzserver'] + ([] if humans else ['-u']) +
                              [world, '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    client = subprocess.Popen(['gzclient'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True)
    rclpy.init()
    node = Spawner()
    print('Waiting for Gazebo ...')
    if not node.cli.wait_for_service(timeout_sec=120.0):
        raise SystemExit('Gazebo did not start; check that the world loads with run_episodes.py --gui.')
    rx, ry, ryaw = robot
    robot_xml = open(ROBOT_SDF).read()
    ok = node.spawn('burger', robot_xml, rx, ry, 0.01, ryaw)
    for name, xml, x, y, z, yaw in items:
        ok &= node.spawn(name, xml, x, y, z, yaw)
    for name, xml, x, y, yaw in people:
        ok &= node.spawn(name, xml, x, y, 0.0, yaw)
    print(f"Spawned the robot and {len(items)} ribbon pieces"
          f"{'' if ok else ' (some failed; see the Gazebo terminal)'}.")
    if humans:
        print('Letting the human figures walk into place ...')
        time.sleep(6.0)                                  # a few seconds of simulation, even at a slow real time factor
        subprocess.run(['gz', 'world', '-p', '1'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('Gazebo is paused. Adjust the view, then use the camera icon in the toolbar for a screenshot '
          '(saved in ~/.gazebo/pictures). Press Ctrl+C here to close.')
    try:
        while client.poll() is None:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        for p in (client, server):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

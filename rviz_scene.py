#!/usr/bin/env python3
"""
Stage a scene in RViz for paper figures (no Gazebo needed).

Publishes the cafe occupancy map, places the TurtleBot3 model, draws the robot's
path and pedestrian routes as coloured lines (solid: crosses, dashed: does not;
colours match make_figures.py) and pedestrians as simple human figures, then
opens RViz with a ready-made view.

Modes:
  routes     all scenario families (the RViz version of the map-routes figure)
  snapshots  one scenario at chosen moments, optionally from a recorded episode

Usage:
  python3 rviz_scene.py routes
  python3 rviz_scene.py routes --view 3d
  python3 rviz_scene.py snapshots --scenario cafe_hs_x_7000 --times 4.5 --robot-time 4.5 --no-lines \\
      --trajectory ~/ask_or_act/runs/validation/always_ask/cafe_hs_x_7000/trajectory.csv

Take the picture with a normal screenshot (Print Screen, or Shift+Print Screen
for an area). Press Ctrl+C here to close.
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import JointState
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_scenarios import FAMILIES, ROBOT, build, eval_seeds   # noqa: E402
from ped_traj import HUMAN_RADIUS, build_trajectory, pose_at        # noqa: E402
from plot_map import read_map_yaml, read_pgm                        # noqa: E402

HOME = os.path.expanduser('~')
MAPS = os.path.join(HOME, 'hunav_ws', 'src', 'hunav_gazebo_wrapper', 'maps')
RVIZ_CFG = os.path.join(HOME, 'ask_or_act', 'worlds', 'scene.rviz')
HEX = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7']
FAM_ORDER = list(dict.fromkeys(f for f, _ in FAMILIES))
rgb = lambda h: tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
FAM_RGB = {f: rgb(HEX[i]) for i, f in enumerate(FAM_ORDER)}
SHORT = {'cc': 'clear_cross', 'cn': 'clear_noncross', 'hs': 'hesitant', 'lt': 'late_turn',
         'sc': 'slow_cross', 'wf': 'wait_far'}


# ------------------------------------------------------------------ scene content
def routes_content(offset_step=0.12):
    lines, people = [], []
    seed = eval_seeds('nominal', 7000, 1)[0]
    for fam, intent in FAMILIES:
        lab = build('cafe', fam, intent, seed, 'nominal')
        dx = offset_step * (FAM_ORDER.index(fam) - (len(FAM_ORDER) - 1) / 2) - lab['route_shift']
        pts = [(p[0] + dx, p[1]) for p in [lab['start']] + lab['waypoints']]
        lines.append((pts, FAM_RGB[fam], intent != 'crossing'))
        if fam == 'wait_far' or (fam == 'hesitant' and intent == 'crossing'):
            people.append((pts[1][0], pts[1][1], math.pi / 2, FAM_RGB[fam], 1.0))
    start = build('cafe', 'clear_cross', 'crossing', seed, 'nominal')['start']
    people.append((start[0], start[1] - 0.45, math.pi / 2, (0.55, 0.55, 0.6), 1.0))
    return lines, people, (ROBOT['start'][0], ROBOT['start'][1], math.pi)


def snapshots_content(scenario, times, robot_time, trajectory=None):
    parts = scenario.split('_')
    family, intent = SHORT[parts[1]], 'crossing' if parts[2] == 'x' else 'not_crossing'
    cond = parts[4] if len(parts) > 4 else 'nominal'
    lab = build(parts[0], family, intent, int(parts[3]), cond)
    if trajectory:
        from stage_scene import from_trajectory
        poses, robot = from_trajectory(trajectory, times, robot_time)
    else:
        traj = build_trajectory(lab['start'], lab['plan'], lab['human_speed'])
        ts = [s[0] for s in traj]
        poses = [pose_at(traj, ts, t - lab['release_delay'])[:3] for t in times]
        rx = ROBOT['start'][0] - ROBOT['speed'] * max(0.0, robot_time - lab['robot_delay'])
        frac = (ROBOT['start'][0] - rx) / (ROBOT['start'][0] - ROBOT['goal'][0])
        robot = (rx, ROBOT['start'][1] + frac * (ROBOT['goal'][1] - ROBOT['start'][1]), math.pi)
    color = FAM_RGB[family]
    n = len(poses)
    people = [(x, y, yaw, color, 1.0 if n == 1 else 0.35 + 0.65 * i / (n - 1)) for i, (x, y, yaw) in enumerate(poses)]
    lines = [([lab['start']] + lab['waypoints'], color, intent != 'crossing')]
    return lines, people, robot


# ------------------------------------------------------------------ ROS node
class Scene(Node):
    def __init__(self, map_name, lines, people, robot, show_lines, human_scale, label_times=None):
        super().__init__('rviz_scene')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/scene_markers', latched)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.tf = StaticTransformBroadcaster(self)
        self.map_pub.publish(self.make_map(map_name))
        self.publish_robot_tf(*robot)
        self.marker_pub.publish(self.make_markers(lines, people, show_lines, human_scale, label_times))
        self.create_timer(0.2, self.publish_joints)

    def make_map(self, name):
        img = read_pgm(os.path.join(MAPS, f'{name}.pgm'))
        res, origin = read_map_yaml(os.path.join(MAPS, f'{name}.yaml'))
        occ = np.full(img.shape, -1, dtype=np.int8)
        occ[img >= 0.75 * 255] = 0          # free (light pixels)
        occ[img <= 0.35 * 255] = 100        # occupied (dark pixels)
        m = OccupancyGrid()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = res
        m.info.width, m.info.height = img.shape[1], img.shape[0]
        m.info.origin.position.x, m.info.origin.position.y = origin[0], origin[1]
        m.info.origin.orientation.w = 1.0
        m.data = np.flipud(occ).flatten().tolist()     # image rows go top-down, the grid bottom-up
        return m

    def publish_robot_tf(self, x, y, yaw):
        now = self.get_clock().now().to_msg()
        a = TransformStamped()
        a.header.stamp, a.header.frame_id, a.child_frame_id = now, 'map', 'odom'
        a.transform.rotation.w = 1.0
        b = TransformStamped()
        b.header.stamp, b.header.frame_id, b.child_frame_id = now, 'odom', 'base_footprint'
        b.transform.translation.x, b.transform.translation.y = x, y
        b.transform.rotation.z, b.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.tf.sendTransform([a, b])

    def publish_joints(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name, js.position = ['wheel_left_joint', 'wheel_right_joint'], [0.0, 0.0]
        self.joint_pub.publish(js)

    def marker(self, mid, mtype, color, alpha=1.0):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = 'scene', mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = (*color, alpha)
        return m

    def make_markers(self, lines, people, show_lines, human_scale, label_times):
        out, k = MarkerArray(), 0
        clear = Marker()
        clear.action = Marker.DELETEALL
        out.markers.append(clear)
        if show_lines:
            path = self.marker(k, Marker.LINE_STRIP, (0.0, 0.0, 0.0))
            path.scale.x = 0.06
            path.points = [Point(x=ROBOT['start'][0], y=ROBOT['start'][1], z=0.02),
                           Point(x=ROBOT['goal'][0], y=ROBOT['goal'][1], z=0.02)]
            out.markers.append(path)
            k += 1
            for pts, color, dashed in lines:
                m = self.marker(k, Marker.LINE_LIST, color)
                m.scale.x = 0.05
                for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                    seg = math.hypot(x1 - x0, y1 - y0)
                    step = 0.3 if dashed else seg
                    on = 0.18 if dashed else seg
                    s = 0.0
                    while s < seg - 1e-6:
                        e = min(s + on, seg)
                        m.points += [Point(x=x0 + (x1 - x0) * s / seg, y=y0 + (y1 - y0) * s / seg, z=0.03),
                                     Point(x=x0 + (x1 - x0) * e / seg, y=y0 + (y1 - y0) * e / seg, z=0.03)]
                        s += step
                out.markers.append(m)
                k += 1
        h = 1.56 * human_scale
        for i, (x, y, yaw, color, alpha) in enumerate(people):
            body = self.marker(k, Marker.CYLINDER, color, alpha)
            body.pose.position.x, body.pose.position.y, body.pose.position.z = x, y, h / 2
            body.scale.x = body.scale.y = 2 * HUMAN_RADIUS * human_scale
            body.scale.z = h
            head = self.marker(k + 1, Marker.SPHERE, (0.9, 0.75, 0.6), alpha)
            head.pose.position.x, head.pose.position.y, head.pose.position.z = x, y, h + 0.12 * human_scale
            head.scale.x = head.scale.y = head.scale.z = 0.24 * human_scale
            out.markers += [body, head]
            k += 2
            if label_times:
                txt = self.marker(k, Marker.TEXT_VIEW_FACING, (0.0, 0.0, 0.0))
                txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = x, y, h + 0.45 * human_scale
                txt.scale.z = 0.22
                txt.text = f't = {label_times[i]:g} s'
                out.markers.append(txt)
                k += 1
        return out


def write_rviz_config(view, center, show_map=True, zoom=70):
    cx, cy = center
    if view == 'top':
        view_cfg = f"""      Class: rviz_default_plugins/TopDownOrtho
      Name: Current View
      Scale: {zoom}
      X: {cx}
      Y: {cy}
      Angle: 0
      Target Frame: <Fixed Frame>"""
    else:
        view_cfg = f"""      Class: rviz_default_plugins/Orbit
      Name: Current View
      Distance: 7.5
      Focal Point:
        X: {cx}
        Y: {cy}
        Z: 0.3
      Pitch: 0.55
      Yaw: 0.35
      Target Frame: <Fixed Frame>"""
    cfg = f"""Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Map
      Name: Map
      Enabled: {'true' if show_map else 'false'}
      Alpha: 1
      Color Scheme: map
      Topic:
        Value: /map
        Depth: 1
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Enabled: true
      Description Source: Topic
      Description Topic:
        Value: /robot_description
    - Class: rviz_default_plugins/MarkerArray
      Name: Scene
      Enabled: true
      Topic:
        Value: /scene_markers
        Depth: 5
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
  Global Options:
    Fixed Frame: map
    Background Color: 255; 255; 255
    Frame Rate: 30
  Views:
    Current:
{view_cfg}
"""
    os.makedirs(os.path.dirname(RVIZ_CFG), exist_ok=True)
    with open(RVIZ_CFG, 'w') as f:
        f.write(cfg)
    return RVIZ_CFG


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['routes', 'snapshots'])
    ap.add_argument('--scenario', default='cafe_hs_x_7000')
    ap.add_argument('--times', nargs='+', type=float, default=[0.0, 2.0, 4.5, 6.5])
    ap.add_argument('--robot-time', type=float, default=4.5)
    ap.add_argument('--trajectory', default=None, help='trajectory.csv of a recorded episode (snapshots)')
    ap.add_argument('--view', choices=['top', '3d'], default='top')
    ap.add_argument('--center', nargs=2, type=float, default=[0.8, -0.6], metavar=('X', 'Y'))
    ap.add_argument('--no-lines', action='store_true', help='leave out the robot path and routes')
    ap.add_argument('--human-scale', type=float, default=1.0)
    ap.add_argument('--label-times', action='store_true', help='show "t = ... s" above each pedestrian (snapshots)')
    ap.add_argument('--map', default='cafe')
    ap.add_argument('--no-map', action='store_true', help='plain white background, no occupancy map')
    ap.add_argument('--zoom', type=float, default=70, help='top view: pixels per metre (larger = closer)')
    args = ap.parse_args()

    if args.mode == 'routes':
        lines, people, robot = routes_content()
    else:
        lines, people, robot = snapshots_content(args.scenario, args.times, args.robot_time, args.trajectory)
    cfg = write_rviz_config(args.view, args.center, not args.no_map, args.zoom)

    env = dict(os.environ, TURTLEBOT3_MODEL='burger')
    procs = [subprocess.Popen(['ros2', 'launch', 'turtlebot3_gazebo', 'robot_state_publisher.launch.py',
                               'use_sim_time:=False'], env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, start_new_session=True)]
    rclpy.init()
    node = Scene(args.map, lines, people, robot, not args.no_lines, args.human_scale,
                 args.times if (args.label_times and args.mode == 'snapshots') else None)
    time.sleep(1.0)
    procs.append(subprocess.Popen(['rviz2', '-d', cfg], env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, start_new_session=True))
    print('RViz is opening. Take a screenshot when it looks right; press Ctrl+C here to close.')
    try:
        while procs[-1].poll() is None:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Straight-line controller with ask-or-act policies, for the Ask-or-Act study.

Drives the TurtleBot3 along a fixed line from its spawn point to a goal,
releases the pedestrians (via /hunav_goal_pose) at a set offset, and, when a
pedestrian is engaged (ahead of the robot and within `engage_dist` along the
line), decides each step whether to GO or STOP according to `policy`:

  always_act, always_cautious, always_ask, threshold, oracle  (see policy.py)

Pedestrian positions are observed at OBS_RATE with Gaussian noise `obs_noise`
(seeded by `noise_seed`, so runs are reproducible). All times are simulation
time, relative to episode start.

Outputs:
  result_file (JSON)     outcome, duration, distances, crossings, asks, answers, stopped time
  traj_file (CSV)        robot and pedestrian positions at every pedestrian update
  decisions_file (CSV)   every decision while a pedestrian is engaged: p, action, reason

Run manually with:
  python3 line_controller.py --ros-args -p use_sim_time:=true -p policy:=threshold
"""
import csv
import json
import math
import os
import random
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from hunav_msgs.msg import Agents
from std_msgs.msg import Int32, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy import POLICIES, AskOrActPolicy, load_predictor   # noqa: E402

HOME = os.path.expanduser('~')
OBS_RATE = 10.0          # Hz, matches build_dataset.py


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class LineController(Node):
    def __init__(self):
        super().__init__('line_controller')
        defaults = dict(
            start_x=2.9, start_y=0.1, goal_x=-2.9, goal_y=0.0,
            speed=0.18, goal_tolerance=0.15, start_delay=2.0, max_duration=90.0,
            human_release_delay=0.0, robot_start_delay=0.0,
            odom_offset_x=0.0, odom_offset_y=0.0,
            result_file='', traj_file='', decisions_file='',
            # decision policy
            policy='always_act',
            model_file=os.path.join(HOME, 'ask_or_act', 'predictor', 'intent_model.pkl'),
            model_name='mlp', calibration='isotonic', calib_condition='nominal',
            p_low=0.2, p_high=0.8,
            engage_dist=1.0,       # m along the line: pedestrians closer than this ahead are engaged
            ask_dist=0.6,          # m ahead: decision deadline for asking
            smooth_n=5,            # predictions averaged before thresholding
            yield_timeout=8.0,     # s of yielding to one pedestrian before falling back to caution
            clear_dist=0.6,        # m past the line: a crossing pedestrian beyond this has cleared
            caution_dist=1.5,      # m from the line, for always_cautious
            answer_timeout=5.0,    # s to wait for an answer before giving up on it
            obs_noise=0.03, noise_seed='', true_intent='')
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        p = lambda n: self.get_parameter(n).value
        self.start, self.goal = (p('start_x'), p('start_y')), (p('goal_x'), p('goal_y'))
        self.speed, self.tol = p('speed'), p('goal_tolerance')
        self.start_delay, self.max_duration = p('start_delay'), p('max_duration')
        self.release_delay, self.robot_delay = p('human_release_delay'), p('robot_start_delay')
        self.offset = (p('odom_offset_x'), p('odom_offset_y'))
        self.result_file, self.traj_file, self.decisions_file = p('result_file'), p('traj_file'), p('decisions_file')
        self.policy = p('policy')
        if self.policy not in POLICIES:
            raise RuntimeError(f'unknown policy {self.policy}; choose from {POLICIES}')
        self.obs_noise = p('obs_noise')
        self.rng = random.Random(f"obs-{p('noise_seed')}")
        model = cal = None
        if self.policy == 'threshold':
            model, cal = load_predictor(p('model_file'), p('model_name'), p('calibration'), p('calib_condition'))
        self.pol = AskOrActPolicy(self.policy, model, cal, p('p_low'), p('p_high'), p('engage_dist'),
                                  p('ask_dist'), p('clear_dist'), p('caution_dist'), p('answer_timeout'),
                                  p('true_intent'), p('smooth_n'), p('yield_timeout'))
        self.method_desc = (self.policy if self.policy != 'threshold' else
                            f"threshold({p('model_name')}, {p('calibration')}@{p('calib_condition')}, "
                            f"band {p('p_low')}-{p('p_high')})")

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.start_pub = self.create_publisher(PoseStamped, '/hunav_goal_pose', 10)
        self.ask_pub = self.create_publisher(Int32, '/robot_ask', 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(Agents, '/human_states', self.on_humans, 10)
        self.create_subscription(String, '/pedestrian_answer', self.on_answer, 10)

        self.pose = None
        self.state = 'WAIT_ODOM'
        self.done = False
        self.t_ready = self.t_start = None
        self.released = False
        self.humans = {}                 # id -> latest true (x, y)
        self.last_obs_t = None
        self.stopped_time = 0.0
        self.last_step_t = None
        self.min_dist, self.min_info = float('inf'), None
        self.crossings, self.line_closest, self.prev_hy, self.traj = [], {}, {}, []
        self.create_timer(0.05, self.step)
        self.get_logger().info(f'Controller ready: {self.method_desc}, goal={self.goal}, speed={self.speed}')

    # ------------------------------------------------------------------ inputs
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        pos = msg.pose.pose.position
        self.pose = (pos.x + self.offset[0], pos.y + self.offset[1], yaw_from_quat(msg.pose.pose.orientation))

    def on_answer(self, msg):
        a = json.loads(msg.data)
        t = self.now() - self.t_start if self.t_start is not None else 0.0
        if self.pol.receive_answer(a['id'], a['intent'], t):
            self.get_logger().info(f"Answer from pedestrian {a['id']}: {a['intent']} (t={t:.2f} s)")

    def on_humans(self, msg):
        for a in msg.agents:
            self.humans[a.id] = (a.position.position.x, a.position.position.y)
        if self.state != 'DRIVING' or self.pose is None:
            return
        t = self.now() - self.t_start
        rx, ry = self.pose[0], self.pose[1]
        for a in msg.agents:
            hx, hy = self.humans[a.id]
            self.traj.append((round(t, 3), round(rx, 3), round(ry, 3), a.id, round(hx, 3), round(hy, 3)))
            d = math.hypot(hx - rx, hy - ry)
            if d < self.min_dist:
                self.min_dist = d
                self.min_info = dict(t=round(t, 3), human=[round(hx, 3), round(hy, 3)], robot=[round(rx, 3), round(ry, 3)])
            if not self.released:
                continue
            best = self.line_closest.get(a.id)
            if best is None or abs(hy) < best['abs_y']:
                self.line_closest[a.id] = dict(t=round(t, 3), x=round(hx, 3), abs_y=round(abs(hy), 3), robot_x=round(rx, 3))
            prev = self.prev_hy.get(a.id)
            if prev is not None and (prev < 0.0) != (hy < 0.0):
                self.crossings.append(dict(human=a.id, t=round(t, 3), x=round(hx, 3), robot_x=round(rx, 3)))
                self.get_logger().info(f'Human {a.id} crossed the line at t={t:.2f} s, x={hx:.2f}; '
                                       f'robot at x={rx:.2f} ({rx - hx:+.2f} m from crossing)')
            self.prev_hy[a.id] = hy

    def observe(self, t_abs):
        """Record noisy pedestrian positions at OBS_RATE (the robot's tracker)."""
        if self.last_obs_t is not None and t_abs - self.last_obs_t < 1.0 / OBS_RATE - 1e-6:
            return
        self.last_obs_t = t_abs
        for hid, (x, y) in self.humans.items():
            self.pol.add_observation(hid, t_abs, x + self.rng.gauss(0.0, self.obs_noise),
                                     y + self.rng.gauss(0.0, self.obs_noise))

    # ------------------------------------------------------------------ output
    def publish_start(self):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = self.goal
        msg.pose.orientation.w = 1.0
        self.start_pub.publish(msg)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def finish(self, outcome, elapsed):
        self.stop()
        self.state = 'DONE'
        md = self.min_dist if self.min_dist < float('inf') else None
        self.get_logger().info(f'Episode finished ({outcome}): duration {elapsed:.2f} s, '
                               f'min distance {md if md is None else round(md, 3)} m, '
                               f'asks {len(self.pol.asks)}, stopped {self.stopped_time:.2f} s')
        if self.result_file:
            res = dict(outcome=outcome, duration=round(elapsed, 3), min_dist=None if md is None else round(md, 3),
                       min_info=self.min_info, crossings=self.crossings,
                       line_closest={str(k): v for k, v in self.line_closest.items()},
                       policy=self.policy, method=self.method_desc, n_asks=len(self.pol.asks), asks=self.pol.asks,
                       answers=self.pol.answers, stopped_time=round(self.stopped_time, 3),
                       speed=self.speed, human_release_delay=self.release_delay, robot_start_delay=self.robot_delay)
            with open(self.result_file, 'w') as f:
                json.dump(res, f, indent=2)
        if self.traj_file:
            with open(self.traj_file, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t', 'robot_x', 'robot_y', 'human_id', 'human_x', 'human_y'])
                w.writerows(self.traj)
        if self.decisions_file:
            with open(self.decisions_file, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t', 'human_id', 'p_crossing', 'action', 'reason', 'ahead_m', 'dist_to_line_m'])
                w.writerows(self.pol.decisions)
        self.done = True

    # -------------------------------------------------------------------- loop
    def step(self):
        if self.pose is None or self.state in ('DONE', 'ABORTED'):
            if self.state == 'DONE':
                self.stop()
            return
        t_abs = self.now()
        if self.humans:
            self.observe(t_abs)

        if self.state == 'WAIT_ODOM':
            off = math.hypot(self.pose[0] - self.start[0], self.pose[1] - self.start[1])
            if off > 0.3:
                self.get_logger().error(f'Robot is at ({self.pose[0]:.2f}, {self.pose[1]:.2f}), {off:.2f} m from '
                                        f'the start point {self.start}. Restart the simulation before each episode.')
                self.state, self.done = 'ABORTED', True
                return
            self.t_ready, self.state = t_abs, 'DELAY'
            self.get_logger().info(f'Robot at ({self.pose[0]:.2f}, {self.pose[1]:.2f}), starting in {self.start_delay:.1f} s')
            return

        if self.state == 'DELAY':
            if t_abs - self.t_ready >= self.start_delay:
                self.t_start, self.state = t_abs, 'DRIVING'
                self.get_logger().info(f'Episode started: humans released in {self.release_delay:.2f} s, '
                                       f'robot starts in {self.robot_delay:.2f} s')
            return

        x, y, yaw = self.pose
        dx, dy = self.goal[0] - x, self.goal[1] - y
        elapsed = t_abs - self.t_start
        dt = 0.0 if self.last_step_t is None else t_abs - self.last_step_t
        self.last_step_t = t_abs
        if not self.released and elapsed >= self.release_delay:
            self.publish_start()
            self.released = True
            self.get_logger().info(f'Humans released at t={elapsed:.2f} s')
        if math.hypot(dx, dy) < self.tol:
            self.finish('reached goal', elapsed)
            return
        if elapsed > self.max_duration:
            self.finish('timed out', elapsed)
            return
        if elapsed < self.robot_delay or self.speed <= 0.0:
            self.stop()
            return
        go, _, to_ask = self.pol.decide(elapsed, x)
        for hid in to_ask:
            self.ask_pub.publish(Int32(data=int(hid)))
            self.get_logger().info(f'Asking pedestrian {hid} (t={elapsed:.2f} s)')
        if not go:
            self.stopped_time += dt
            self.stop()
            return
        err = wrap(math.atan2(dy, dx) - yaw)
        cmd = Twist()
        cmd.angular.z = max(-1.0, min(1.0, 1.5 * err))
        cmd.linear.x = 0.0 if abs(err) > 0.5 else self.speed
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = LineController()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()

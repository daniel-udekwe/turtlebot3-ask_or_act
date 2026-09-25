#!/usr/bin/env python3
"""
Straight-line controller for the Ask-or-Act study.

Drives the TurtleBot3 along a fixed line from its spawn point to a goal,
releases the HuNavSim humans (via /hunav_goal_pose) at a set offset, and
records the episode (all times in simulation time, relative to episode start):

  * result_file (JSON): outcome, duration, minimum robot-human distance,
    line crossings, each human's closest approach to the robot's line
  * traj_file (CSV):    robot and human positions at every human update

It exits by itself when the episode ends, so it can be scripted.

Run manually with:
  python3 line_controller.py --ros-args -p use_sim_time:=true
Calibration (robot stays still): add  -p speed:=0.0 -p max_duration:=25.0
"""
import csv
import json
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from hunav_msgs.msg import Agents


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class LineController(Node):
    def __init__(self):
        super().__init__('line_controller')
        defaults = dict(
            start_x=2.9, start_y=0.1,          # expected spawn point, checked before starting
            goal_x=-2.9, goal_y=0.0,
            speed=0.18,                        # m/s (Burger max ~0.22); 0 for calibration runs
            goal_tolerance=0.15,               # m
            start_delay=2.0,                   # s of sim time before the episode starts
            max_duration=90.0,                 # s of sim time before timeout
            human_release_delay=0.0,           # s after episode start to release humans
            robot_start_delay=0.0,             # s after episode start before the robot moves
            odom_offset_x=0.0, odom_offset_y=0.0,  # added to /odom to get world coordinates
            result_file='', traj_file='')
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        p = lambda n: self.get_parameter(n).value
        self.start = (p('start_x'), p('start_y'))
        self.goal = (p('goal_x'), p('goal_y'))
        self.speed = p('speed')
        self.tol = p('goal_tolerance')
        self.start_delay = p('start_delay')
        self.max_duration = p('max_duration')
        self.release_delay = p('human_release_delay')
        self.robot_delay = p('robot_start_delay')
        self.offset = (p('odom_offset_x'), p('odom_offset_y'))
        self.result_file = p('result_file')
        self.traj_file = p('traj_file')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.start_pub = self.create_publisher(PoseStamped, '/hunav_goal_pose', 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(Agents, '/human_states', self.on_humans, 10)

        self.pose = None                     # (x, y, yaw) in world frame
        self.state = 'WAIT_ODOM'
        self.done = False
        self.t_ready = self.t_start = None
        self.released = False
        self.min_dist = float('inf')
        self.min_info = None                 # closest approach details
        self.crossings = []                  # line crossings
        self.line_closest = {}               # per human: closest approach to the robot's line
        self.prev_hy = {}
        self.traj = []
        self.create_timer(0.05, self.step)   # 20 Hz, follows sim time
        self.get_logger().info(f'Line controller ready, goal={self.goal}, speed={self.speed}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        pos = msg.pose.pose.position
        self.pose = (pos.x + self.offset[0], pos.y + self.offset[1],
                     yaw_from_quat(msg.pose.pose.orientation))

    def on_humans(self, msg):
        if self.state != 'DRIVING' or self.pose is None:
            return
        t = self.now() - self.t_start
        rx, ry = self.pose[0], self.pose[1]
        for a in msg.agents:
            hx, hy = a.position.position.x, a.position.position.y
            self.traj.append((round(t, 3), round(rx, 3), round(ry, 3), a.id, round(hx, 3), round(hy, 3)))
            d = math.hypot(hx - rx, hy - ry)
            if d < self.min_dist:
                self.min_dist = d
                self.min_info = dict(t=round(t, 3), human=[round(hx, 3), round(hy, 3)],
                                     robot=[round(rx, 3), round(ry, 3)])
            if not self.released:
                continue
            best = self.line_closest.get(a.id)
            if best is None or abs(hy) < best['abs_y']:
                self.line_closest[a.id] = dict(t=round(t, 3), x=round(hx, 3), abs_y=round(abs(hy), 3),
                                               robot_x=round(rx, 3))
            prev = self.prev_hy.get(a.id)
            if prev is not None and (prev < 0.0) != (hy < 0.0):
                self.crossings.append(dict(human=a.id, t=round(t, 3), x=round(hx, 3), robot_x=round(rx, 3)))
                self.get_logger().info(f'Human {a.id} crossed the line at t={t:.2f} s, x={hx:.2f}; '
                                       f'robot at x={rx:.2f} ({rx - hx:+.2f} m from crossing)')
            self.prev_hy[a.id] = hy

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
        self.get_logger().info(f'Episode finished ({outcome}): duration {elapsed:.2f} s, min robot-human '
                               f'distance {md:.2f} m' if md is not None else
                               f'Episode finished ({outcome}): duration {elapsed:.2f} s, no humans seen')
        if self.min_info:
            self.get_logger().info(f"Closest approach at t={self.min_info['t']:.2f} s: human "
                                   f"{self.min_info['human']}, robot {self.min_info['robot']}")
        if self.result_file:
            res = dict(outcome=outcome, duration=round(elapsed, 3),
                       min_dist=round(md, 3) if md is not None else None, min_info=self.min_info,
                       crossings=self.crossings,
                       line_closest={str(k): v for k, v in self.line_closest.items()},
                       speed=self.speed, human_release_delay=self.release_delay,
                       robot_start_delay=self.robot_delay)
            with open(self.result_file, 'w') as f:
                json.dump(res, f, indent=2)
        if self.traj_file:
            with open(self.traj_file, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t', 'robot_x', 'robot_y', 'human_id', 'human_x', 'human_y'])
                w.writerows(self.traj)
        self.done = True

    def step(self):
        if self.pose is None or self.state in ('DONE', 'ABORTED'):
            if self.state == 'DONE':
                self.stop()
            return
        t = self.now()

        if self.state == 'WAIT_ODOM':
            off = math.hypot(self.pose[0] - self.start[0], self.pose[1] - self.start[1])
            if off > 0.3:
                self.get_logger().error(
                    f'Robot is at ({self.pose[0]:.2f}, {self.pose[1]:.2f}), {off:.2f} m from the start '
                    f'point {self.start}. Restart the simulation and respawn the robot before each episode.')
                self.state = 'ABORTED'
                self.done = True
                return
            self.t_ready = t
            self.state = 'DELAY'
            self.get_logger().info(f'Robot at ({self.pose[0]:.2f}, {self.pose[1]:.2f}), '
                                   f'starting in {self.start_delay:.1f} s')
            return

        if self.state == 'DELAY':
            if t - self.t_ready >= self.start_delay:
                self.t_start = t
                self.state = 'DRIVING'
                self.get_logger().info(f'Episode started: humans released in {self.release_delay:.2f} s, '
                                       f'robot starts in {self.robot_delay:.2f} s')
            return

        # DRIVING
        x, y, yaw = self.pose
        dx, dy = self.goal[0] - x, self.goal[1] - y
        dist = math.hypot(dx, dy)
        elapsed = t - self.t_start
        if not self.released and elapsed >= self.release_delay:
            self.publish_start()
            self.released = True
            self.get_logger().info(f'Humans released at t={elapsed:.2f} s')
        if dist < self.tol:
            self.finish('reached goal', elapsed)
            return
        if elapsed > self.max_duration:
            self.finish('timed out', elapsed)
            return
        if elapsed < self.robot_delay or self.speed <= 0.0:
            self.stop()
            return
        err = wrap(math.atan2(dy, dx) - yaw)
        cmd = Twist()
        cmd.angular.z = max(-1.0, min(1.0, 1.5 * err))
        cmd.linear.x = 0.0 if abs(err) > 0.5 else self.speed   # turn in place if badly misaligned
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

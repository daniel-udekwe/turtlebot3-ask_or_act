#!/usr/bin/env python3
"""
Straight-line controller for the Ask-or-Act study.

Drives the TurtleBot3 along a fixed line from its spawn point to a goal,
publishes the start signal that releases the HuNavSim humans, and reports
episode duration and minimum robot-human distance (all in simulation time).

Run with:
  python3 line_controller.py --ros-args -p use_sim_time:=true
"""
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
        # Goal and motion parameters (world / map frame)
        self.declare_parameter('goal_x', -2.9)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('speed', 0.18)            # m/s, Burger max is ~0.22
        self.declare_parameter('goal_tolerance', 0.15)   # m
        self.declare_parameter('start_delay', 2.0)       # s of sim time before starting
        self.declare_parameter('max_duration', 90.0)     # s of sim time before timeout
        # Offset added to /odom to get world coordinates (0,0 if odom == world)
        self.declare_parameter('odom_offset_x', 0.0)
        self.declare_parameter('odom_offset_y', 0.0)

        p = lambda n: self.get_parameter(n).value
        self.goal = (p('goal_x'), p('goal_y'))
        self.speed = p('speed')
        self.tol = p('goal_tolerance')
        self.start_delay = p('start_delay')
        self.max_duration = p('max_duration')
        self.offset = (p('odom_offset_x'), p('odom_offset_y'))

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.start_pub = self.create_publisher(PoseStamped, '/hunav_goal_pose', 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(Agents, '/human_states', self.on_humans, 10)

        self.pose = None          # (x, y, yaw) in world frame
        self.state = 'WAIT_ODOM'
        self.t_ready = None
        self.t_start = None
        self.min_dist = float('inf')
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
        for a in msg.agents:
            d = math.hypot(a.position.position.x - self.pose[0],
                           a.position.position.y - self.pose[1])
            self.min_dist = min(self.min_dist, d)

    def publish_start(self):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = self.goal
        msg.pose.orientation.w = 1.0
        self.start_pub.publish(msg)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def step(self):
        if self.pose is None:
            return
        t = self.now()

        if self.state == 'WAIT_ODOM':
            self.t_ready = t
            self.state = 'DELAY'
            self.get_logger().info(f'Robot at ({self.pose[0]:.2f}, {self.pose[1]:.2f}), '
                                   f'starting in {self.start_delay:.1f} s')
            return

        if self.state == 'DELAY':
            if t - self.t_ready >= self.start_delay:
                self.publish_start()
                self.t_start = t
                self.state = 'DRIVING'
                self.get_logger().info('Episode started: humans released, robot driving')
            return

        if self.state == 'DRIVING':
            x, y, yaw = self.pose
            dx, dy = self.goal[0] - x, self.goal[1] - y
            dist = math.hypot(dx, dy)
            elapsed = t - self.t_start

            if dist < self.tol or elapsed > self.max_duration:
                self.stop()
                self.state = 'DONE'
                outcome = 'reached goal' if dist < self.tol else 'timed out'
                md = f'{self.min_dist:.2f} m' if self.min_dist < float('inf') else 'n/a'
                self.get_logger().info(f'Episode finished ({outcome}): duration {elapsed:.2f} s, '
                                       f'min robot-human distance {md}')
                return

            err = wrap(math.atan2(dy, dx) - yaw)
            cmd = Twist()
            cmd.angular.z = max(-1.0, min(1.0, 1.5 * err))
            # Turn in place if badly misaligned, otherwise drive forward
            cmd.linear.x = 0.0 if abs(err) > 0.5 else self.speed
            self.cmd_pub.publish(cmd)
            return

        if self.state == 'DONE':
            self.stop()


def main():
    rclpy.init()
    node = LineController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Scripted pedestrians for the Ask-or-Act study.

Spawns one Gazebo model per pedestrian and moves it along its deterministic
trajectory (ped_traj.py) in simulation time, starting when the release signal
arrives on /hunav_goal_pose (published by line_controller.py). Publishes:

  /human_states        hunav_msgs/Agents   pedestrian poses, same topic line_controller.py uses
  /pedestrian_markers  MarkerArray         body cylinder and intent label, for RViz

Run with:
  python3 scripted_pedestrians.py --ros-args -p use_sim_time:=true -p scenario_id:=cafe_cc_x_00
"""
import json
import math
import os
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from hunav_msgs.msg import Agent, Agents
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ped_traj import HUMAN_RADIUS, build_trajectory, pose_at   # noqa: E402

HOME = os.path.expanduser('~')
HEIGHT = 1.7

HUMAN_SDF = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="pedestrian">
    <static>true</static>
    <link name="body">
      <collision name="collision">
        <pose>0 0 {HEIGHT / 2:.2f} 0 0 0</pose>
        <geometry><cylinder><radius>{HUMAN_RADIUS}</radius><length>{HEIGHT}</length></cylinder></geometry>
        <surface><contact><collide_without_contact>true</collide_without_contact></contact></surface>
      </collision>
      <visual name="torso">
        <pose>0 0 0.78 0 0 0</pose>
        <geometry><cylinder><radius>{HUMAN_RADIUS}</radius><length>1.56</length></cylinder></geometry>
        <material><ambient>0.2 0.4 0.8 1</ambient><diffuse>0.2 0.4 0.8 1</diffuse></material>
      </visual>
      <visual name="head">
        <pose>0 0 1.68 0 0 0</pose>
        <geometry><sphere><radius>0.12</radius></sphere></geometry>
        <material><ambient>0.9 0.75 0.6 1</ambient><diffuse>0.9 0.75 0.6 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


def quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class ScriptedPedestrians(Node):
    def __init__(self):
        super().__init__('scripted_pedestrians')
        self.declare_parameter('labels_file', os.path.join(HOME, 'ask_or_act', 'scenarios', 'labels.json'))
        self.declare_parameter('scenario_id', '')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('rate', 20.0)
        sid = self.get_parameter('scenario_id').value
        labels = {l['id']: l for l in json.load(open(self.get_parameter('labels_file').value))}
        if sid not in labels:
            raise RuntimeError(f'scenario_id "{sid}" not found in labels file')
        self.label = labels[sid]
        self.frame = self.get_parameter('frame_id').value

        # One pedestrian per scenario for now; the lists allow more later
        self.peds = [dict(id=1, name='pedestrian_1',
                          traj=build_trajectory(self.label['start'], self.label['plan'],
                                                self.label['human_speed']))]
        for p in self.peds:
            p['times'] = [s[0] for s in p['traj']]

        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.set_cli = self.create_client(SetEntityState, '/set_entity_state')
        self.state_pub = self.create_publisher(Agents, '/human_states', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/pedestrian_markers', 10)
        self.create_subscription(PoseStamped, '/hunav_goal_pose', self.on_release, 10)

        self.t_release = None
        self.spawn_futures = None
        self.pending_set = None
        self.ready = False
        self.create_timer(1.0 / self.get_parameter('rate').value, self.step)
        self.get_logger().info(f"Scenario {sid}: {self.label['family']} / {self.label['intent']}")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_release(self, _msg):
        if self.t_release is None:
            self.t_release = self.now()
            self.get_logger().info('Pedestrians released')

    def spawn(self):
        if not self.spawn_cli.service_is_ready():
            return
        self.spawn_futures = []
        for p in self.peds:
            x, y, yaw, _ = pose_at(p['traj'], p['times'], 0.0)
            req = SpawnEntity.Request()
            req.name = p['name']
            req.xml = HUMAN_SDF
            req.initial_pose.position.x, req.initial_pose.position.y = x, y
            (req.initial_pose.orientation.x, req.initial_pose.orientation.y,
             req.initial_pose.orientation.z, req.initial_pose.orientation.w) = quat(yaw)
            self.spawn_futures.append(self.spawn_cli.call_async(req))

    def step(self):
        if not self.ready:
            if self.spawn_futures is None:
                self.spawn()
            elif all(f.done() for f in self.spawn_futures) and self.set_cli.service_is_ready():
                self.ready = True
                self.get_logger().info('Pedestrians spawned, waiting for release')
            return

        t_rel = self.now() - self.t_release if self.t_release is not None else 0.0
        stamp = self.get_clock().now().to_msg()
        agents = Agents()
        agents.header.stamp, agents.header.frame_id = stamp, self.frame
        markers = MarkerArray()
        for p in self.peds:
            x, y, yaw, v = pose_at(p['traj'], p['times'], t_rel)
            qx, qy, qz, qw = quat(yaw)

            # Move the Gazebo model (skip if the previous request is still in flight)
            if self.pending_set is None or self.pending_set.done():
                req = SetEntityState.Request()
                req.state = EntityState(name=p['name'], reference_frame='world')
                req.state.pose.position.x, req.state.pose.position.y = x, y
                (req.state.pose.orientation.x, req.state.pose.orientation.y,
                 req.state.pose.orientation.z, req.state.pose.orientation.w) = qx, qy, qz, qw
                self.pending_set = self.set_cli.call_async(req)

            a = Agent()
            a.id = p['id']
            a.position.position.x, a.position.position.y = x, y
            (a.position.orientation.x, a.position.orientation.y,
             a.position.orientation.z, a.position.orientation.w) = qx, qy, qz, qw
            a.velocity.linear.x, a.velocity.linear.y = v * math.cos(yaw), v * math.sin(yaw)
            for field, val in (('yaw', yaw), ('linear_vel', v), ('radius', HUMAN_RADIUS), ('name', p['name'])):
                if hasattr(a, field):
                    setattr(a, field, val)
            agents.agents.append(a)

            body = Marker()
            body.header.stamp, body.header.frame_id = stamp, self.frame
            body.ns, body.id, body.type = 'pedestrian', p['id'] * 10, Marker.CYLINDER
            body.pose.position.x, body.pose.position.y, body.pose.position.z = x, y, HEIGHT / 2
            body.pose.orientation.w = 1.0
            body.scale.x = body.scale.y = 2 * HUMAN_RADIUS
            body.scale.z = HEIGHT
            body.color.r, body.color.g, body.color.b, body.color.a = 0.2, 0.4, 0.8, 0.8
            label = Marker()
            label.header = body.header
            label.ns, label.id, label.type = 'pedestrian', p['id'] * 10 + 1, Marker.TEXT_VIEW_FACING
            label.pose.position.x, label.pose.position.y, label.pose.position.z = x, y, HEIGHT + 0.3
            label.pose.orientation.w = 1.0
            label.scale.z = 0.25
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f"{self.label['family']} ({self.label['intent']})"
            markers.markers += [body, label]

        self.state_pub.publish(agents)
        self.marker_pub.publish(markers)


def main():
    rclpy.init()
    node = ScriptedPedestrians()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import baxter_interface
from gevent.server import StreamServer
from mprpc import RPCServer
import numpy as np
import rospy
#import torch

import multiprocessing as mp
import signal
import sys
import time

def vec2flat(v):
  # TODO
  pass

def joint2flat(d, keys):
  arr = []
  for key in keys:
    arr.append(d[key])
  return np.array(arr)

def flat2joint(arr, keys):
  d = {}
  for idx, key in enumerate(keys):
    d[key] = arr[idx]
  return d

class PolicyDriverServer(RPCServer):
  def poll_transition(self):
    # TODO
    info = {
        "obs": None,
        "res": 0.0,
        "init": False,
        "done": False,
    }
    pass

  def post_transition(self, obs, res, init, done):
    # TODO
    print("DEBUG: driver server: post transition")
    pass

class BaxterArmEnv(object):
  def __init__(self, limb, jointvel_action_clip):
    self.limb = limb
    self.jointvel_action_clip = jointvel_action_clip

    self.prev_theta = None
    self.prev_theta_dot = None
    self.curr_theta = None
    self.curr_theta_dot = None

  def reset(self):
    joint_keys = self.limb.joint_names()
    pos = limb.endpoint_pose()["position"]
    theta = joint2flat(self.limb.joint_angles(), joint_keys)
    theta_dot = np.zeros(theta.shape, dtype=theta.dtype)
    prev_theta = np.array(theta)
    prev_theta_dot = np.array(theta_dot)
    t = rospy.Time.now()
    self.prev_t = t
    self.prev_theta = prev_theta
    self.prev_theta_dot = prev_theta_dot
    self.curr_t = t
    self.curr_theta = theta
    self.curr_theta_dot = theta_dot

  def step(self, action):
    joint_keys = self.limb.joint_names()

    if action is not None:
      assert self.jointvel_action_clip is not None
      assert self.jointvel_action_clip >= 0.0
      ctrl_theta_dot = np.clip(action, -self.jointvel_action_clip, self.jointvel_action_clip)
      self.limb.set_joint_velocities(flat2joint(ctrl_theta_dot, joint_keys))

    t = rospy.Time.now()
    delta_t = (t - self.curr_t).to_sec()

    pos = limb.endpoint_pose()["position"]
    theta = joint2flat(self.limb.joint_angles(), joint_keys)
    theta_dot = (theta - self.curr_theta) / delta_t

    self.prev_t = self.curr_t
    self.prev_theta = self.curr_theta
    self.prev_theta_dot = self.curr_theta_dot
    self.curr_t = t
    self.curr_theta = theta
    self.curr_theta_dot = theta_dot

  def get_obs(self):
    raise NotImplementedError

  def get_reward(self):
    raise NotImplementedError

  def done(self):
    raise NotImplementedError

class BaxterReachingEnv(BaxterArmEnv):
  def __init__(self, *args):
    super().__init__(self, *args)
    self.tg_box = None

  def reset(self):
    # TODO: randomly initialize a target region.
    super().reset(self)

  def get_obs(self):
    # TODO
    obs = np.concatenate((self.curr_theta, self.curr_theta_dot, self.prev_theta, self.prev_theta_dot), axis=0)
    return obs

def main():
  def sigint_handler(signal, frame):
    # TODO
    sys.exit(0)
  signal.signal(signal.SIGINT, sigint_handler)

  rospy.init_node("baxter_policy_driver_node")
  time.sleep(1.0)

  # Constants.
  STEP_FREQ = 20.0   # Hz
  SAFE_JOINTVEL_ACTION_CLIP = 0.5

  arm = "right"
  #arm = "left"

  limb = baxter_interface.Limb(arm)

  # TODO: setup the policy.

  # TODO
  env = BaxterArmEnv(limb, SAFE_JOINTVEL_ACTION_CLIP)

  # TODO: episodic control.
  loop_rate = rospy.Rate(STEP_FREQ)
  while True:
    loop_rate.sleep()

  driver = PolicyDriverServer()
  driver.post_transition(None, None, None, None)

  server = StreamServer(("127.0.0.1", 12345), driver)
  server.serve_forever()

if __name__ == "__main__":
  main()

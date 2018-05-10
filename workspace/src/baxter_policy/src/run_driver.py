#!/usr/bin/env python

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import baxter_interface
from gevent.server import StreamServer
from mprpc import RPCServer
import numpy as np
import rospy
import torch

import multiprocessing as mp
import signal
import sys
import time

def pos2flat(v):
  return np.array([v.x, v.y, v.z])

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

class DummyPolicy(object):
  def __init__(self):
    pass

  def execute(self, obs):
    return None

class TorchPolicy(object):
  def __init__(self, pol_params, pol_fn):
    self.pol_params = pol_params
    self.pol_fn = pol_fn

  def set_flat_param(self, flat_param):
    # TODO
    #deserialize_vars(self.pol_params, flat_param)
    pass

  def execute(self, obs):
    obs_var = torch.unsqueeze(torch.from_numpy(obs), 0)
    act_var = self.pol_fn(obs_var)
    act = act_var.data.numpy()[0,:]
    return act

class BaxterArmState(object):
  def __init__(self, limb, jointvel_action_clip):
    self.limb = limb
    self.jointvel_action_clip = jointvel_action_clip

    #self.prev_t = None
    #self.prev_theta = None
    #self.prev_theta_dot = None
    self.curr_t = None
    self.curr_pos = None
    self.curr_theta = None
    self.curr_theta_dot = None

  def reset(self):
    joint_keys = self.limb.joint_names()

    t = rospy.Time.now()

    pos = pos2flat(limb.endpoint_pose()["position"])
    theta = joint2flat(self.limb.joint_angles(), joint_keys)
    theta_dot = np.zeros(theta.shape, dtype=theta.dtype)

    #prev_theta = np.array(theta)
    #prev_theta_dot = np.array(theta_dot)
    #self.prev_t = t
    #self.prev_theta = prev_theta
    #self.prev_theta_dot = prev_theta_dot
    self.curr_t = t
    self.curr_pos = pos
    self.curr_theta = theta
    self.curr_theta_dot = theta_dot

  def step_up(self, action):
    joint_keys = self.limb.joint_names()

    if action is not None:
      assert self.jointvel_action_clip is not None
      assert self.jointvel_action_clip >= 0.0
      ctrl_theta_dot = np.clip(action, -self.jointvel_action_clip, self.jointvel_action_clip)
      self.limb.set_joint_velocities(flat2joint(ctrl_theta_dot, joint_keys))

  def step_down(self):
    joint_keys = self.limb.joint_names()

    t = rospy.Time.now()
    delta_t = (t - self.curr_t).to_sec()

    pos = pos2flat(limb.endpoint_pose()["position"])
    theta = joint2flat(self.limb.joint_angles(), joint_keys)
    theta_dot = (theta - self.curr_theta) / delta_t

    #self.prev_t = self.curr_t
    #self.prev_theta = self.curr_theta
    #self.prev_theta_dot = self.curr_theta_dot
    self.curr_t = t
    self.curr_pos = pos
    self.curr_theta = theta
    self.curr_theta_dot = theta_dot

class BaxterReachingEnv(object):
  def __init__(self, state, step_freq):
    self.state = state
    self.step_freq = step_freq
    self.loop_rate = rospy.Rate(2.0 * STEP_FREQ)
    self.tg_box = None

  def reset(self):
    # TODO: randomly initialize a target region.
    self.state.reset()
    obs = self.get_obs()
    return obs

  def step(self, action):
    self.state.step_up(action)
    self.loop_rate.sleep()
    self.state.step_down()
    # TODO: get obs, reward, etc.
    obs = self.get_obs()
    res = 0.0
    done = False
    return obs, res, done, None

  def sleep(self):
    self.loop_rate.sleep()

  def get_obs(self):
    # TODO
    obs = np.concatenate((self.curr_theta, self.curr_theta_dot, self.curr_pos), axis=0)
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
  SAFE_JOINTVEL_ACTION_CLIP = 0.1
  #SAFE_JOINTVEL_ACTION_CLIP = 0.5

  arm = "right"
  #arm = "left"

  limb = baxter_interface.Limb(arm)

  # TODO: setup the policy.
  policy = DummyPolicy()
  #policy = TorchPolicy(...)

  # TODO
  state = BaxterArmState(limb, SAFE_JOINTVEL_ACTION_CLIP)
  env = BaxterReachingEnv(state, STEP_FREQ)

  # TODO
  policy = TorchPolicy(None, None)

  # TODO: episodic control.
  obs = env.reset()
  act = policy.execute(obs)
  while True:
    obs, res, done, _ = env.step(act)
    if done:
      break
    act = policy.execute(obs)
    env.sleep()

  driver = PolicyDriverServer()
  driver.post_transition(None, None, None, None)

  server = StreamServer(("127.0.0.1", 12345), driver)
  server.serve_forever()

if __name__ == "__main__":
  main()

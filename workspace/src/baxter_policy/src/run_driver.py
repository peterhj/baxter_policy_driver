#!/usr/bin/env python

#import msgpack
#import msgpack_numpy
#msgpack_numpy.patch()

from env import *
from models import *
from utils import *

import baxter_interface
#from gevent.server import StreamServer
#from mprpc import RPCServer
import numpy as np
import rospy
import torch

import multiprocessing as mp
import signal
import sys
import time

if False:
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

class RealtimeTrajectory(object):
  def __init__(self):
    self.init_obs = None
    self.steps = []

  def reset(self):
    self.init_obs = None
    del self.steps[:]

  def rollout(self, env, policy, horizon=None):
    self.reset()
    obs = env.reset()
    self.init_obs = obs
    act = policy.execute(obs)
    while True:
      obs, res, done, _ = env.step(act)
      print("DEBUG: step: pos:", obs[-3:])
      self.steps.append((act, res, done, obs))
      if done:
        break
      elif horizon is not None and len(self.steps) >= horizon:
        break
      act = policy.execute(obs)
      env.sleep()
    env.stop()

class DummyPolicy(object):
  def __init__(self):
    pass

  def execute(self, obs):
    return None

class RandomNormalPolicy(object):
  def __init__(self, act_dim):
    self.act_shape = (act_dim,)

  def execute(self, obs):
    return np.random.normal(scale=0.16, size=self.act_shape)

class TorchPolicy(object):
  def __init__(self, act_dim, pol_params, pol_fn):
    self.act_dim = act_dim
    self.act_shape = (act_dim,)
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

def main():
  def sigint_handler(signal, frame):
    # TODO
    print("DEBUG: got ctrl-C, exiting...")
    sys.exit(0)
  signal.signal(signal.SIGINT, sigint_handler)

  rospy.init_node("baxter_policy_driver_node")
  time.sleep(1.0)

  ARM = "right"
  #ARM = "left"

  # Constants.
  ACTION_DIM = 7
  STEP_FREQ = 20.0   # Hz
  SAFE_JOINTVEL_ACTION_CLIP = 0.25

  limb = baxter_interface.Limb(ARM)

  # TODO: setup the policy.
  policy = DummyPolicy()
  policy = RandomNormalPolicy(ACTION_DIM)
  #policy = TorchPolicy(...)

  # TODO
  state = BaxterArmState(limb, STEP_FREQ, SAFE_JOINTVEL_ACTION_CLIP)
  env = BaxterReachingEnv(state)

  # TODO: episodic control.
  traj = RealtimeTrajectory()
  print("DEBUG: starting episode...")
  traj.rollout(env, policy, horizon=100)
  print("DEBUG: terminating episode...")

  return

  driver = PolicyDriverServer()
  driver.post_transition(None, None, None, None)

  server = StreamServer(("127.0.0.1", 12345), driver)
  server.serve_forever()

if __name__ == "__main__":
  main()

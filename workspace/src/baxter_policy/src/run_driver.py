#!/usr/bin/env python

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from env import *
from memory import *
from models import *
from utils import *

import baxter_interface
import numpy as np
from redis import StrictRedis
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

class TorchDeterministicPolicy(object):
  def __init__(self, act_dim, pol_params, pol_fn):
    self.act_dim = act_dim
    self.act_shape = (act_dim,)
    self.pol_params = pol_params
    self.pol_fn = pol_fn

  def set_flat_param(self, flat_param):
    deserialize_vars(self.pol_params, flat_param)
    pass

  def execute(self, obs):
    obs_var = const_var(torch.unsqueeze(torch.from_numpy(obs).type(torch.FloatTensor), 0))
    act_var = self.pol_fn(obs_var)
    act = act_var.data.numpy()[0,:]
    return act

class TorchGaussianPolicy(object):
  def __init__(self, act_dim, pol_params, pol_fn):
    self.act_dim = act_dim
    self.act_shape = (act_dim,)
    self.pol_params = pol_params
    self.pol_fn = pol_fn

  def set_flat_param(self, flat_param):
    deserialize_vars(self.pol_params, flat_param)
    pass

  def execute(self, obs):
    obs_var = const_var(torch.unsqueeze(torch.from_numpy(obs).type(torch.FloatTensor), 0))
    act_dist_var = self.pol_fn(obs_var).view(1, 2, self.act_dim)
    act_mean_var = act_dist_var[:,0,:]
    act_logstd_var = act_dist_var[:,1,:]
    act_std_var = torch.exp(act_logstd_var)
    act_var = torch.normal(act_mean_var, act_std_var)
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
  SAFE_ACTION_CLIP = 0.16
  #SAFE_ACTION_CLIP = 0.25
  SAFE_STEP_HORIZON = 100

  # TODO
  limb = baxter_interface.Limb(ARM)
  state = BaxterArmState(limb, STEP_FREQ, SAFE_ACTION_CLIP)
  env = BaxterReachingEnv(state)

  pol_params, pol_init_fns, pol_fn = build_ddpg_gaussian_policy_fn(env.obs_shape()[0], ACTION_DIM)
  param_sz = flat_count_vars(pol_params)
  print("DEBUG: policy param sz:", param_sz)
  #serialize_vars(pol_params)

  # TODO: setup the policy.
  #policy = DummyPolicy()
  #policy = RandomNormalPolicy(ACTION_DIM)
  #policy = TorchDeterministicPolicy(ACTION_DIM, pol_params, pol_fn)
  policy = TorchGaussianPolicy(ACTION_DIM, pol_params, pol_fn)

  #replay = EpisodicReplayMemory(1000)
  #client = RPCClient("127.0.0.1", 12345)
  client = StrictRedis(host="127.0.0.1", port=12345, db=0)

  traj_count = 0
  last_epoch = None

  assert client.flushdb()

  while True:
    # TODO: fetch the latest version of the policy params.
    if False:
      while True:
        msg = client.get("pol_flat_param")
        if msg is None:
          time.sleep(0.01)
          continue
        pol_flat_param, epoch = msgpack.unpackb(msg)
        if epoch is not None:
          if last_epoch is not None and last_epoch >= epoch:
            time.sleep(0.01)
            continue
          last_epoch = epoch
          break
      deserialize_vars(pol_params, pol_flat_param)
    else:
      while True:
        msg = client.get("pol_epoch")
        if msg is None:
          time.sleep(0.01)
          continue
        epoch = msgpack.unpackb(msg)
        if epoch is not None:
          if last_epoch is not None and last_epoch >= epoch:
            time.sleep(0.01)
            continue
          last_epoch = epoch
          break

    traj = RealtimeTrajectory()
    print("DEBUG: starting episode {}".format(traj_count))
    traj.rollout(env, policy, horizon=SAFE_STEP_HORIZON)
    #print("DEBUG: terminating episode...")
    traj_count += 1

    p_obs, p_act, p_res, p_done = traj.vectorize()
    #print "DEBUG: packed obs:", p_obs
    #print "DEBUG: packed act:", p_act
    #print np.amax(p_act)
    #print np.amin(p_act)
    #print np.sum(p_res)
    #replay.append_traj(p_obs, p_act, p_res, p_done)

    # TODO: send the current trajectory experience.
    msg = msgpack.packb((p_obs, p_act, p_res, p_done, last_epoch))
    client.set("traj_mem:{}".format(last_epoch), msg)

    time.sleep(1.0)

  return

  driver = PolicyDriverServer()
  driver.post_transition(None, None, None, None)

  server = StreamServer(("127.0.0.1", 12345), driver)
  server.serve_forever()

if __name__ == "__main__":
  main()

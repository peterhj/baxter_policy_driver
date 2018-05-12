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

  # Robot constants.
  ACTION_DIM = 7
  STEP_FREQ = 20.0   # Hz
  #SAFE_ACTION_CLIP = 0.16
  SAFE_ACTION_CLIP = 0.20
  #SAFE_ACTION_CLIP = 0.25
  SAFE_STEP_HORIZON = 100

  # Learning constants.
  MANUAL_RESET = False
  #MANUAL_RESET = True
  EVAL_INTERVAL = 10

  limb = baxter_interface.Limb(ARM)
  state = BaxterArmState(limb, STEP_FREQ, SAFE_ACTION_CLIP)
  env = BaxterReachingEnv(state)

  #hidden_dims = [400, 300]
  hidden_dims = [128, 128]

  pol_params, pol_init_fns, pol_fn = build_ddpg_gaussian_policy_fn(env.obs_shape()[0], ACTION_DIM, hidden_dims)
  param_sz = flat_count_vars(pol_params)
  print("DEBUG: policy param sz:", param_sz)
  #serialize_vars(pol_params)

  # TODO: setup the policy.
  #policy = DummyPolicy()
  #policy = RandomNormalPolicy(ACTION_DIM)
  #policy = TorchDeterministicPolicy(ACTION_DIM, pol_params, pol_fn)
  policy = TorchGaussianPolicy(ACTION_DIM, pol_params, pol_fn)

  #eval_policy = TorchDeterministicPolicy(ACTION_DIM, pol_params, pol_fn)

  #replay = EpisodicReplayMemory(1000)
  #client = RPCClient("127.0.0.1", 12345)
  client = StrictRedis(host="127.0.0.1", port=12345, db=0)

  traj_count = 0
  last_epoch = None

  assert client.flushdb()

  print("DEBUG: driver: ready")
  while True:
    while True:
      #print("DEBUG: driver: waiting for pol flat param...")
      msg = client.get("pol_flat_param")
      if msg is None:
        time.sleep(0.5)
        continue
      pol_flat_param, epoch = msgpack.unpackb(msg)
      if epoch is not None:
        if last_epoch is not None and last_epoch >= epoch:
          time.sleep(0.5)
          continue
        last_epoch = epoch
        break
    deserialize_vars(pol_params, torch.from_numpy(pol_flat_param))

    if False:
      if traj_count > 0 and traj_count % EVAL_INTERVAL == 0:
        print("DEBUG: eval: episodes: {}".format(traj_count))
        eval_traj = RealtimeTrajectory()
        eval_traj.rollout(env, eval_policy, horizon=SAFE_STEP_HORIZON)
        print("DEBUG: eval:   ret: {} success: {}".format(eval_traj.sum_return(), env.success))

    if MANUAL_RESET:
      # TODO
      pass

    traj = RealtimeTrajectory()
    traj.rollout(env, policy, horizon=SAFE_STEP_HORIZON)
    print("DEBUG: train: episode: {} ret: {} distance: {} success: {}".format(traj_count, traj.sum_return(), env.get_distance(), env.success))
    traj_count += 1

    p_obs, p_act, p_res, p_done = traj.vectorize()
    msg = msgpack.packb((p_obs, p_act, p_res, p_done, last_epoch))
    client.set("traj_mem:{}".format(last_epoch), msg)

    time.sleep(0.5)

if __name__ == "__main__":
  main()

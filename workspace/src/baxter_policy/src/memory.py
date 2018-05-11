#!/usr/bin/env python

import numpy as np

from collections import deque

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
      #print("DEBUG: step: pos:", obs[-6:-3], "tg pos:", obs[-3:], "res:", res, "done:", done)
      self.steps.append((act, res, done, obs))
      if done:
        break
      elif horizon is not None and len(self.steps) >= horizon:
        break
      act = policy.execute(obs)
      env.sleep()
    env.stop()

  def vectorize(self):
    assert len(self.steps) >= 1
    packed_obs = np.zeros((len(self.steps) + 1, self.init_obs.shape[0]))
    packed_act = np.zeros((len(self.steps), self.steps[0][0].shape[0]))
    packed_res = np.zeros(len(self.steps))
    packed_done = np.zeros(len(self.steps))
    packed_obs[0,:] = self.init_obs
    for k in range(len(self.steps)):
      packed_obs[k+1,:] = self.steps[k][3]
      packed_act[k,:] = self.steps[k][0]
      packed_res[k] = self.steps[k][1]
      packed_done[k] = 1.0 if self.steps[k][2] else 0.0
    return packed_obs, packed_act, packed_res, packed_done

class EpisodicReplayMemory(object):
  def __init__(self, capacity):
    self.capacity = capacity
    self.cache = deque([], capacity)

  def count(self):
    return len(self.cache)

  def append_traj(self, packed_obs, packed_act, packed_res, packed_done):
    self.cache.append((packed_obs, packed_act, packed_res, packed_done))

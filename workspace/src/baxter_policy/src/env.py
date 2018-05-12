#!/usr/bin/env python

from utils import *

import numpy as np
import rospy

class BaxterArmState(object):
  def __init__(self, limb, step_freq, jointvel_clip):
    self.limb = limb
    self.step_freq = step_freq
    self.jointvel_clip = jointvel_clip

    #self.prev_t = None
    #self.prev_theta = None
    #self.prev_theta_dot = None
    self.curr_t = None
    self.curr_pos = None
    self.curr_theta = None
    self.curr_theta_dot = None

  def step_frequency(self):
    return self.step_freq

  def reset(self):
    cmd_delay = max(0.0, min(0.2, 2.0 / self.step_freq))
    self.limb.set_command_timeout(cmd_delay)

    # TODO: enable this only with other safety wrappers.
    #self.limb.set_joint_position_speed(1.0)

    joint_keys = self.limb.joint_names()

    t = rospy.Time.now()

    pos = pos2flat(self.limb.endpoint_pose()["position"])
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
      assert self.jointvel_clip is not None
      assert self.jointvel_clip >= 0.0
      ctrl_theta_dot = np.clip(action, -self.jointvel_clip, self.jointvel_clip)
      self.limb.set_joint_velocities(flat2joint(ctrl_theta_dot, joint_keys))

  def step_down(self):
    joint_keys = self.limb.joint_names()

    t = rospy.Time.now()
    delta_t = (t - self.curr_t).to_sec()

    pos = pos2flat(self.limb.endpoint_pose()["position"])
    theta = joint2flat(self.limb.joint_angles(), joint_keys)
    theta_dot = (theta - self.curr_theta) / delta_t

    #self.prev_t = self.curr_t
    #self.prev_theta = self.curr_theta
    #self.prev_theta_dot = self.curr_theta_dot
    self.curr_t = t
    self.curr_pos = pos
    self.curr_theta = theta
    self.curr_theta_dot = theta_dot

  def stop(self):
    self.limb.exit_control_mode()

class Box(object):
  def __init__(self, x_bounds, y_bounds, z_bounds):
    assert x_bounds[0] <= x_bounds[1]
    assert y_bounds[0] <= y_bounds[1]
    assert z_bounds[0] <= z_bounds[1]
    self.x_bounds = np.array(x_bounds)
    self.y_bounds = np.array(y_bounds)
    self.z_bounds = np.array(z_bounds)

  def center(self):
    xc = np.mean(self.x_bounds)
    yc = np.mean(self.y_bounds)
    zc = np.mean(self.z_bounds)
    return np.array([xc, yc, zc])

  def contains(self, p):
    if p[0] < self.x_bounds[0] or p[0] > self.x_bounds[1]:
      return False
    if p[1] < self.y_bounds[0] or p[1] > self.y_bounds[1]:
      return False
    if p[2] < self.z_bounds[0] or p[2] > self.z_bounds[1]:
      return False
    return True

class BaxterReachingEnv(object):
  def __init__(self, state, horizon=None):
    self.state = state
    self.horizon = horizon
    self.step_freq = state.step_frequency()
    self.loop_rate = rospy.Rate(2.0 * self.step_freq)
    # TODO: these are specific to the right arm.
    #valid_x = [0.3, 0.65]
    valid_x = [0.5, 0.9]
    #valid_y = [-0.75, -0.25]
    valid_y = [-0.8, 0.0]
    #valid_z = [0.25, 0.75]
    valid_z = [0.25, 0.65]
    self.valid_box = Box(valid_x, valid_y, valid_z)
    # TODO: static target box for testing.
    #target_x = [0.425, 0.525]
    target_x = [0.65, 0.75]
    #target_y = [-0.55, -0.45]
    target_y = [-0.45, -0.35]
    #target_z = [0.45, 0.55]
    target_z = [0.4, 0.5]
    #self.tg_box = None
    self.tg_box = Box(target_x, target_y, target_z)

    self.k = 0
    self.prev_dist_to_tg = None
    self.success = False

  def reset(self):
    # TODO: randomly initialize a target region.
    self.state.reset()
    obs = self.get_obs()
    self.k = 0
    #self.prev_dist_to_tg = None
    #self.prev_dist_to_tg = 0.0
    dx = self.state.curr_pos - self.tg_box.center()
    #self.prev_dist_to_tg = np.linalg.norm(dx)
    self.prev_dist_to_tg = dx.dot(dx)
    self.success = False
    return obs

  def step(self, action):
    self.state.step_up(action)
    self.loop_rate.sleep()
    self.state.step_down()
    obs = self.get_obs()
    dx = self.state.curr_pos - self.tg_box.center()
    #dist_to_tg = np.linalg.norm(dx)
    dist_to_tg = dx.dot(dx)
    reached = dist_to_tg <= 0.05
    out_of_bounds = not self.valid_box.contains(self.state.curr_pos)
    res = (self.prev_dist_to_tg - dist_to_tg) * self.step_freq
    res -= 0.1 * action.dot(action)
    done = reached
    if self.horizon is not None:
      done = done or self.k >= self.horizon
    if reached:
      #assert not out_of_bounds
      res += 100.0
      self.success = True
    #elif out_of_bounds:
    #  res -= 0.1
    #  #res -= 1.0
    self.k += 1
    self.prev_dist_to_tg = dist_to_tg
    return obs, res, done, None

  def sleep(self):
    self.loop_rate.sleep()

  def stop(self):
    self.state.stop()

  def obs_shape(self):
    return (20,)

  def get_obs(self):
    # TODO
    obs = np.concatenate((self.state.curr_theta, self.state.curr_theta_dot, self.state.curr_pos, self.tg_box.center()), axis=0)
    return obs

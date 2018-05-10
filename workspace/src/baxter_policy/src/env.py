#!/usr/bin/env python

from utils import *

import numpy as np
import rospy

class BaxterArmState(object):
  def __init__(self, limb, step_freq, jointvel_action_clip):
    self.limb = limb
    self.step_freq = step_freq
    self.jointvel_action_clip = jointvel_action_clip

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
      assert self.jointvel_action_clip is not None
      assert self.jointvel_action_clip >= 0.0
      ctrl_theta_dot = np.clip(action, -self.jointvel_action_clip, self.jointvel_action_clip)
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

class BaxterReachingEnv(object):
  def __init__(self, state):
    self.state = state
    self.step_freq = state.step_frequency()
    self.loop_rate = rospy.Rate(2.0 * self.step_freq)
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

  def stop(self):
    self.state.stop()

  def get_obs(self):
    # TODO
    obs = np.concatenate((self.state.curr_theta, self.state.curr_theta_dot, self.state.curr_pos), axis=0)
    return obs

#!/usr/bin/env python

import baxter_interface
import numpy as np
import rospy

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

def main():
  def sigint_handler(signal, frame):
    # TODO
    sys.exit(0)
  signal.signal(signal.SIGINT, sigint_handler)

  rospy.init_node("baxter_policy_test_node")
  time.sleep(1.0)

  arm = "right"
  #arm = "left"
  print("DEBUG: arm:     {}".format(arm))

  limb = baxter_interface.Limb(arm)

  joint_keys = limb.joint_names()
  pos = pos2flat(limb.endpoint_pose()["position"])
  theta = joint2flat(limb.joint_angles(), joint_keys)

  print("DEBUG: joints:  {}".format(joint_keys))
  print("DEBUG: end pos: {}".format(pos))
  print("DEBUG: theta:   {}".format(theta))

if __name__ == "__main__":
  main()

#!/usr/bin/env python

import numpy as np

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

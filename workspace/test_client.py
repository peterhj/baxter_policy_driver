#!/usr/bin/env python

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from mprpc import RPCClient
import numpy as np

def main():
  client = RPCClient("127.0.0.1", 12345)
  obs, act, res, done = (np.ones((10, 10)), np.zeros((9, 5)), np.zeros(9), np.zeros(9))
  #print isinstance(obs, np.ndarray)
  client.call("put", "traj_mem", (obs, act, res, done))
  mem = client.call("get", "traj_mem")
  print mem

if __name__ == "__main__":
  main()

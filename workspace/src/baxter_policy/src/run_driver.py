#!/usr/bin/env python

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from gevent.server import StreamServer
from mprpc import RPCServer
import numpy as np

import multiprocessing as mp

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

def main():
  driver = PolicyDriverServer()
  driver.post_transition(None, None, None, None)

  server = StreamServer(("127.0.0.1", 12345), driver)
  server.serve_forever()

if __name__ == "__main__":
  main()

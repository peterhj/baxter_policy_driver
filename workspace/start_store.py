#!/usr/bin/env python

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from gevent.server import StreamServer
from mprpc import RPCServer

class BaxterPolicyStore(RPCServer):
  def __init__(self):
    super(BaxterPolicyStore, self).__init__(self)
    self.kvdata = {}

  def put(self, key, value):
    self.kvdata[key] = value

  def get(self, key):
    if key in self.kvdata:
      return self.kvdata[key]
    return None

def main():
  server = StreamServer(("127.0.0.1", 12345), BaxterPolicyStore())
  server.serve_forever()

if __name__ == "__main__":
  main()

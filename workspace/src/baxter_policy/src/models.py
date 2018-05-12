#!/usr/bin/env python

import numpy as np
import torch
from torch.autograd import Variable

from functools import reduce
from operator import mul

def var(data):
  return Variable(data, requires_grad=True, volatile=False)

def const_var(data):
  return Variable(data, requires_grad=False, volatile=False)

def flat_count_vars(vars_):
  offset = 0
  for v in vars_:
    v_dim = v.data.size()
    flat_len = reduce(mul, list(v_dim), 1)
    offset += flat_len
  return offset

def deserialize_vars(vars_, src_data):
  offset = 0
  for v in vars_:
    v_dim = v.data.size()
    flat_len = reduce(mul, list(v_dim), 1)
    #v.data.resize_(flat_len)
    #v.data.copy_(src_data[offset:offset+flat_len])
    #v.data.resize_(v_dim)
    v.data.view(flat_len).copy_(src_data[offset:offset+flat_len])
    offset += flat_len
  return offset

def serialize_vars(vars_, dst_data):
  offset = 0
  for v in vars_:
    v_dim = v.data.size()
    flat_len = reduce(mul, list(v_dim), 1)
    #v.data.resize_(flat_len)
    #dst_data[offset:offset+flat_len].copy_(v.data)
    #v.data.resize_(v_dim)
    dst_data[offset:offset+flat_len].copy_(v.data.view(flat_len))
    offset += flat_len
  return offset

class Params(object):
  def __init__(self):
    self._keys = []
    self._kvs = {}

  def get_var(self, key, init_fn):
    if key not in self._kvs:
      self._keys.append(key)
      self._kvs[key] = (var(init_fn()), init_fn)
    v, _ = self._kvs[key]
    return v

  def build(self):
    param_vars = []
    param_init_fns = []
    for key in reversed(self._keys):
      v, init_fn = self._kvs[key]
      param_vars.append(v)
      param_init_fns.append(init_fn)
    return param_vars, param_init_fns

def build_normc_linear_init(kernel_dim, dtype=None, std=1.0):
  def init_fn():
    x = np.random.randn(*kernel_dim).astype(np.float32)
    x *= std / np.sqrt(np.square(x).sum(axis=1, keepdims=True))
    return torch.from_numpy(x).type(dtype)
  return init_fn

def build_ddpg_policy_fn(obs_dim, act_dim, hidden_dims=[400, 300], dtype=torch.FloatTensor):
  h1, h2 = hidden_dims
  params = Params()
  def pol_fn(obs):
    x = obs
    a1 = params.get_var("a1", build_normc_linear_init((h1, obs_dim), dtype))
    b1 = params.get_var("b1", lambda: torch.zeros(h1).type(dtype))
    x = torch.nn.functional.linear(x, a1, b1)
    x = torch.nn.functional.relu(x)
    a2 = params.get_var("a2", build_normc_linear_init((h2, h1), dtype))
    b2 = params.get_var("b2", lambda: torch.zeros(h2).type(dtype))
    x = torch.nn.functional.linear(x, a2, b2)
    x = torch.nn.functional.relu(x)
    a_out = params.get_var("a_out", build_normc_linear_init((act_dim, h2), std=0.01, dtype=dtype))
    b_out = params.get_var("b_out", lambda: torch.zeros(act_dim).type(dtype))
    x = torch.nn.functional.linear(x, a_out, b_out)
    return x
  pol_fn(const_var(torch.zeros(1, obs_dim).type(dtype)))
  param_vars, param_init_fns = params.build()
  return param_vars, param_init_fns, pol_fn

def build_ddpg_gaussian_policy_fn(obs_dim, act_dim, hidden_dims=[400, 300], dtype=torch.FloatTensor):
  h1, h2 = hidden_dims
  params = Params()
  def pol_fn(obs):
    x = obs
    a1 = params.get_var("a1", build_normc_linear_init((h1, obs_dim), dtype))
    b1 = params.get_var("b1", lambda: torch.zeros(h1).type(dtype))
    x = torch.nn.functional.linear(x, a1, b1)
    x = torch.nn.functional.relu(x)
    a2 = params.get_var("a2", build_normc_linear_init((h2, h1), dtype))
    b2 = params.get_var("b2", lambda: torch.zeros(h2).type(dtype))
    x = torch.nn.functional.linear(x, a2, b2)
    x = torch.nn.functional.relu(x)
    a_mean = params.get_var("a_mean", build_normc_linear_init((act_dim, h2), std=0.01, dtype=dtype))
    b_mean = params.get_var("b_mean", lambda: torch.zeros(act_dim).type(dtype))
    x_mean = torch.nn.functional.linear(x, a_mean, b_mean)
    a_logstd = params.get_var("a_logstd", build_normc_linear_init((act_dim, h2), std=0.01, dtype=dtype))
    b_logstd = params.get_var("b_logstd", lambda: torch.zeros(act_dim).type(dtype))
    x_logstd = torch.nn.functional.linear(x, a_logstd, b_logstd)
    x = torch.cat((x_mean, x_logstd), 1)
    return x
  pol_fn(const_var(torch.zeros(1, obs_dim).type(dtype)))
  param_vars, param_init_fns = params.build()
  return param_vars, param_init_fns, pol_fn

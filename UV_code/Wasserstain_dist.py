# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 08:59:11 2019

@author: Administrator
"""
import torch as t
import torch.nn.functional as F

def one_hot(labels, batch_size):
    index = t.eye(batch_size).cuda()
    one_hot = t.index_select(index, dim = 0, index = labels)
    return one_hot

def distance(s_feature, t_feature):
    m = t_feature.size()[0]
    n = s_feature.size()[0]
    a = s_feature.pow(2).sum(1).unsqueeze(1).repeat(1,m)
    b = t_feature.pow(2).sum(1).repeat(n,1)
    distance = (a + b -2*(s_feature.mm(t_feature.t()))).pow(0.5)
    return distance 

class Wasserstein(object):
  """Class to hold (ref to) data and compute Wasserstein distance."""
  def __init__(self, input_s, input_t, weight_matrix, basedist):
    """Inits Wasserstein with source and target data."""
    self.source_bs = input_s.shape[0]
    self.target_bs = input_t.shape[0]
    self.lamda = 0.1
    self.input_s = input_s
    self.input_t = input_t
    self.weight_matrix = weight_matrix
    if basedist is None:
      basedist = self.l2dist
    self.basedist = basedist 
   
  def l2dist(self, source, target, weight):
    """Computes pairwise Euclidean distances in tensorflow."""
    def flatten_batch(x):
      dim = t.prod(t.tensor(x.shape[1:]))
      return x.reshape([-1, dim])
    def scale_batch(x):
      dim = t.prod(t.tensor(x.shape[1:]))
      return x/dim.double().sqrt()
    def prepare_batch(x):
      return scale_batch(flatten_batch(x))

    target_flat = prepare_batch(target)  # shape: [bs, nt]
    target_sqnorms = t.sum(target_flat.pow(2),1).unsqueeze(1)
    target_sqnorms_t = target_sqnorms.transpose(0,1)  # shape: [bs, nt]

    source_flat = prepare_batch(source)  # shape: [bs, ns]
    source_sqnorms = t.sum(source_flat.pow(2),1).unsqueeze(1)

    dotprod = source_flat.mm(target_flat.transpose(0,1))
    sqdist = source_sqnorms - 2*dotprod + target_sqnorms_t #broadcast
    dist = F.relu(sqdist).sqrt()  # potential tiny negatives are suppressed
    return dist * weight  # shape: [ns, nt]


  def grad_hbar(self, v, source_bs, reuse=True):
    """Compute gradient of hbar function for Wasserstein iteration."""
    source_ims = self.input_s
    target_data = self.input_t

    c_xy = self.basedist(source_ims, target_data, self.weight_matrix)
    c_xy = c_xy - v  # [gradbs, trnsize]
    idx = t.argmin(c_xy, dim=1) # [1] (index of subgradient)
    target_bs = self.target_bs
    xi_ij = one_hot(idx, target_bs)  # find matches, [gradbs, trnsize]
    xi_ij = xi_ij.mean(dim=0).unsqueeze(0)  # [1, trnsize]
    grad = 1./target_bs - xi_ij  # output: [1, trnsize]
    return grad

  def hbar(self, v, reuse=True):
    """Compute value of hbar function for Wasserstein iteration."""
    source_ims = self.input_s
    target_data = self.input_t

    c_xy = self.basedist(source_ims, target_data, self.weight_matrix)
    c_avg = c_xy.mean()
    c_xy = c_xy - c_avg
    c_xy = c_xy - v

    c_xy_min,_ = t.min(c_xy, dim=1)  # min_y[ c(x, y) - v(y) ]
    c_xy_min = t.mean(c_xy_min)     # expectation wrt x
    return t.mean(v, dim=1) + c_xy_min + c_avg # avg wrt y

  def k_step(self, k, v, vt, c, reuse=True):
    """Perform one update step of Wasserstein computation."""
    grad_h = self.grad_hbar(vt, self.source_bs, reuse=reuse)
    vt = vt + c/k.sqrt()*grad_h
    v = ((k-1.)*v + vt)/k
    return k+1, v, vt, c

  def gamma_cal(self, v, reuse = True):
    ''' Compute the gamma matrix.'''
    source_ims = self.input_s
    target_data = self.input_t
    source_bs = self.source_bs
    target_bs = self.target_bs
    c_xy = self.basedist(source_ims, target_data, self.weight_matrix)
    k = t.exp(-self.lamda * c_xy)
    u = c_xy.mean(dim=0)/t.reshape(k.mm(v.transpose()),[target_bs])
    gamma = t.mm(t.diag(t.reshape(u,[source_bs])).mm(k), t.diag(t.reshape(v,[target_bs])))
    return gamma

  def match_cal(self, v, reuse = True):
    ''' Compute the match matrix.'''
    source_ims = self.input_s
    target_data = self.input_t

    c_xy = self.basedist(source_ims, target_data, self.weight_matrix)
    c_avg = t.mean(c_xy)
    c_xy = c_xy - c_avg
    c_xy = c_xy - v
    match = t.argmin(c_xy, dim=1)  # min_y[ c(x, y) - v(y) ]   
    return match, c_xy

  def dist(self, C=.1, nsteps=20, reset=False):
    """Compute Wasserstein distance (Alg.2 in [Genevay etal, NIPS'16])."""
    target_bs = self.target_bs
    vtilde = t.zeros([1, target_bs]).cuda()
    v = t.zeros([1, target_bs]).cuda()
    k = t.tensor(1.).cuda() # restart averaging from 1 in each call

    # (unrolled) optimization loop. first iteration, create variables
   
    while (k < nsteps):   
        k, v, vtilde, C = self.k_step(k, v, vtilde, C)

    v_new = v.detach() # only transmit gradient through cost
    val = self.hbar(v_new)
    #gamma = self.gamma_cal(v)
    #match, dis_match = self.match_cal(v)
    return val.mean()




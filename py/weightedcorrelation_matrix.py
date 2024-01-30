# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 00:15:32 2021

@author: Daniel
"""
import pandas as pd
import numpy as np
from numpy import random

def weighted_covariance(weights,signal, signal2):
 weighted_meanx= np.sum(weights*signal)/np.sum(weights)*np.ones(len(weights))
 weighted_meany= np.sum(weights*signal2)/np.sum(weights)*np.ones(len(weights))
 data = weights*(signal-weighted_meanx)*(signal2-weighted_meany)
 return np.sum(data)

def correlation_matrix(features,weights,signal):
  K=len(features)
  correlation= np.empty((K, K), dtype=float)
  for i in features:
    for j in features:
      XY=weighted_covariance(weights,signal[i], signal[j])
      XX=weighted_covariance(weights,signal[i], signal[i])
      YY=weighted_covariance(weights,signal[j], signal[j])
      R=XY/(np.sqrt(XX*YY))
      correlation[features.index(i)][features.index(j)]=R
  return correlation
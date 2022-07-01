# -*- coding: utf-8 -*-
"""
This function is used to calculate degree of membership function for DENFIS (eqn 8). 
All the memebership functions are triangular functions depending upon three 

parameters: 
     data_in-- input data
     cluster.cls --- a matrix of cluster centers
     Dthr -- the threshold value for the evolving clustering method (ECM), between 0 and 1. 
     d --- a parameter for the width of the triangular membership function.
     
@author: Manish Kakar
"""
import numpy as np

def calcDegreeMF(data_in, cluster_cls, d, Dthr):
    num_dt = len(data_in)
    num_cls = len(cluster_cls)
    #cluster_c = cluster_cls[:,0:2]
    cluster_c = np.delete(cluster_cls,-1,1)
    miu_rule_matrix = np.zeros([num_dt, num_cls])
    for i in range(num_dt):
       for j in range(num_cls):
          a = cluster_c[j] - d * Dthr  # b = cluster_c[j]
          cc = cluster_c[j] + d * Dthr #
          left = (data_in[i] - a)/(cluster_c[j] - a) # x = data_tst[i]
          right = (cc - data_in[i])/(cc - cluster_c[j])	
          temp_min = np.min([left,right], axis=0)
          temp_max = np.max([temp_min, np.zeros(len(temp_min))],axis = 0)
          temp = np.prod(temp_max) #min, max comp
          miu_rule_matrix[i, j] = temp
          
    return(miu_rule_matrix)

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:07:53 2019
cls_denorm = min_Y + cls_norm*(max_Y - min_Y) 
@author: kmi
"""
import numpy as np

def denorm_data(dt_norm, range_data, min_scale,max_scale) :
   row_data = len(dt_norm)
   col_data = np.shape(dt_norm)[1]
   data_denorm = np.zeros([row_data, col_data])
   for j in range(col_data) :
      min_data = range_data[0, j]
      max_data = range_data[1, j]
      for i in range(row_data): 
         data_denorm[i, j] = min_data + (dt_norm[i, j] - min_scale) * (max_data - min_data)/(max_scale - min_scale)
   return(data_denorm)

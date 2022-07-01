# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:24:29 2019

@author: kmi
"""
import numpy as np

def norm_data(OData, range_data, min_scale,max_scale) :
   row_data = len(OData)
   col_data = np.shape(OData)[1]
   data_norm = np.zeros([row_data, col_data])
   for j in range(col_data):
       min_data = range_data[0, j]
       max_data = range_data[1, j]
       for i in range(row_data):
           data_norm[i, j] = min_scale + (OData[i, j] - min_data)*(max_scale - min_scale)/(max_data - min_data)
   return(data_norm)
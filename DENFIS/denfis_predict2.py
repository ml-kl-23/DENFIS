# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 08:51:05 2019

@author: kmi
"""
import numpy as np

from CalcMF_DENFIS import calcDegreeMF
from Defuzzifier import defuzzifier
from Denorm_Data import denorm_data
from Norm_Data import norm_data

def add_one(number):
    return number + 1

def denfis_predict(ModelInput, NewData) :

    mod = ModelInput
    model = mod.copy()
    
    data_test = NewData
    data_test2 = data_test.copy()
    
    min_scale = 0
    max_scale = 1
    
    cluster_c = model['cls']
    cluster_c2 = cluster_c.copy()
    
    func_tsk_a = model['func_tsk']
    func_tsk_b = func_tsk_a.copy()
    
    range_data_ori = model['range_data']
    range_data_ori2 = range_data_ori.copy()
    
    Dthr = model['Dthr']
    d = model['d']
    
    cluster_c3 = norm_data(cluster_c2, range_data_ori2, min_scale, max_scale)
    data_test3 = norm_data(data_test2, np.delete(range_data_ori2,-1,1),min_scale, max_scale)
    #data_test4 = data_test3.copy()
    
    num_cls = len(cluster_c3)
    num_dt = len(data_test3)
    num_inpvar = np.shape(data_test3)[1]
    
    #temp = np.zeros([num_cls, num_inpvar])
    miu_rule = np.zeros([num_dt,num_cls])
    miu_rule = calcDegreeMF(data_test3, cluster_c3, d, Dthr)
    
    
    range_output = np.zeros((2, 1))
    range_output[1] = 1
    
    #defuz4 = np.zeros([num_dt,1])
    defuz4 = defuzzifier(data = data_test3, rule = None, range_output = range_output, 
        names_varoutput = None, varout_mf = None, miu_rule = miu_rule, 
        type_defuz = None, type_model = "TSK", func_tsk = func_tsk_b)
    
    n_of_cols = np.shape(range_data_ori2)[1]
    
    range_output2 = np.zeros((2, 1))
    range_output2[0] = range_data_ori2[0,n_of_cols-1]
    range_output2[1] = range_data_ori2[1,n_of_cols-1]
    
    res = denorm_data(dt_norm=defuz4, range_data= range_output2, min_scale = 0, max_scale= 1)

    return(res)

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:35:42 2019

@author: kmi
"""

import numpy as np
import pandas as pd    
from tqdm import tqdm
    
from ECM import ECM_CLUST
from CalcMF_DENFIS import calcDegreeMF
from Defuzzifier import defuzzifier
from Denorm_Data import denorm_data
from Norm_Data import norm_data


def add_one(number):
    return number + 1


def DENFIS(data_train, range_data, Dthr, max_iter, step_size, d ):   
  
  
    data_train1 = data_train[0:len(data_train)] # training data
    ### Normalize the training data
    data_train2 = norm_data(OData=data_train1, range_data= range_data, min_scale=0,max_scale=1)
    ### make copies as arrays ar mutable
    data_train3 = data_train2.copy()
    data_train4 = data_train3.copy()
    data_train5 = data_train4.copy()
    
    
    ############  Cluster centers from ECM
    Dthr2 = Dthr
    cluster_cls = ECM_CLUST(ts = data_train2 , Dthr = Dthr2)
    cluster_cls2 = cluster_cls.copy()
    num_cls = len(cluster_cls)
    
    ### Prepare the data for input to the fuzzifier
    data_train_minusLastCol=np.delete(data_train3,-1,1)
    data_train_minusLastCol2 = data_train_minusLastCol.copy()
    dt_input = data_train_minusLastCol2
    
    ######  Calculate degree of MF for the cluster centers
    ### Fuzzify the cluster centers     
    d2 = d
    miu_rule = calcDegreeMF(data_train_minusLastCol2, cluster_cls=cluster_cls2, d = d2, Dthr= Dthr2) 
    miu_rule2 = miu_rule.copy()
    
    ##########  Defuzzify cluster centers ###############################
    ### Define constants
#    max_iter =100
#    alpha = 0.01
#    Dthr = 0.2
#    d = 2
    alpha = step_size
    defuz = np.zeros([len(data_train4),1])  
    range_output = np.zeros((2, 1))
    range_output[1] = 1
    num_dt = len(data_train2)
    num_inpvar = np.shape(data_train4)[1] - 1
    
    last_col = np.asarray([data_train4[:,-1]])
    last_col = last_col.T
    
#    
    ##create randomly uniformly distributed cluster centers unifromly distributed as consequent func_tsk
   
    #np.random.seed(101)
    #rand_uniform = np.random.uniform(0,1,num_ele) #Create unoiform distribution samples
    
    num_ele = num_cls*(num_inpvar +1)
    rand_uniform = np.random.random(num_ele) #Create unoiform distribution samples
    func_tsk = np.reshape(rand_uniform, ((num_cls,num_inpvar+1)))
    func_tsk2 = func_tsk.copy() 
    
    ## for Testing with specific values of func_tsk
#    func_tsk = pd.read_csv("func_tsk_Seed101_Clust3.csv", header = None)  
#    func_tsk = func_tsk.values
#    func_tsk2 = func_tsk.copy() 
    
    range_output = np.zeros((2, 1))
    range_output[1] = 1
    func_tsk_new = []
    
    ### Progress bar ######
    with tqdm(total=max_iter) as pbar:

   
        for i in range(max_iter):
           
           ### Calculate defuzzification
           defuz = defuzzifier(data = dt_input , rule = None, range_output = range_output, names_varoutput = None, 
                       varout_mf = None, miu_rule = miu_rule2, type_defuz = None,  type_model = "TSK", 
                       func_tsk = func_tsk)
           defuz2 = defuz.copy()
           
           ### Calculate the error 
           gal1 = defuz2 - last_col
           
           func_tsk_var = func_tsk2[:, :-1]
           func_tsk_var2 = func_tsk_var.copy()   
    
           func_tsk_cont = func_tsk[:,[np.shape(func_tsk2)[1]-1]]
           #func_tsk_cont2 =  func_tsk_cont.copy()
    
           for ii in range(num_dt):
              gal = defuz2[ii] - data_train5[ii,-1]	  
              sum_miu = np.sum([miu_rule[ii]])
              
              for mm in range(len(func_tsk_cont)):
                  
                   if (sum_miu != 0): 
                     func_tsk_cont[mm] =  func_tsk_cont[mm] - alpha *gal * (miu_rule[ii, mm]/sum_miu)	       
                    
                   else: 
                    func_tsk_cont[mm] =  func_tsk_cont[mm] - alpha * gal* miu_rule[ii, mm]
                   
           #print(func_tsk_cont)         
           func_tsk_new = np.hstack((func_tsk_var2, func_tsk_cont))
           func_tsk = func_tsk_new	
        	
           cluster_cls2 = denorm_data(dt_norm = cluster_cls, range_data=range_data, min_scale=0, max_scale=1)
           pbar.update(1)
           pbar.ncols =100
           pbar.set_description("DENFIS progress: " )
          
        mod = {'cls' : cluster_cls2, 
               'func_tsk' : func_tsk_new,
               'range_data' : range_data, 
               'Dthr' : Dthr, 
               'd' : d, 
               "type_model" : "CLUSTERING"}

#
    return(mod)
    pbar.close()    
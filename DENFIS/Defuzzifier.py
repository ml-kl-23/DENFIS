# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:29:13 2019
USAGE :

import numpy as np
import pandas as pd
from itertools import chain    
    
from ECM import ECM_CLUST
from ProgressBar import update_progress
from CalcMF_DENFIS import calcDegreeMF

dataframe2 = pd.read_csv('GasFurnace292pts.csv', usecols = [1,2,3])
ts2 = dataframe2.values
#ts2 = ts2[0:len(ts2),1:4]
ts = np.array(ts2)

min_range = np.min(ts,axis=0)
max_range = np.max(ts,axis=0)
range_data = np.asarray([min_range.T, max_range.T])

data_train = ts[0:204] # training data
data_train2 = data_train.copy() ## Make a copy for passing in the function ECM
data_train_minusLastCol = ts[0:204,0:2]
data_train_minusLastCol2 = data_train_minusLastCol.copy()
dt_input = data_train_minusLastCol2
data = dt_input

## from other files
cluster_cls = ECM_CLUST(ts = data_train2 , Dthr=0.2)
miu_rule = calcDegreeMF(data_train_minusLastCol2, cluster_cls, d=2, Dthr=0.2) 

range_output = np.zeros((2, 1))
range_output[1] = 1

func_tsk = pd.read_csv("func_tsk2.csv", header = None)  
func_tsk = func_tsk.values 
  
    
a = defuzzifier(data = dt_input , rule = None, range_output = range_output, names_varoutput = None, 
               varout_mf = None, miu_rule = miu_rule, type_defuz = None,  type_model = "TSK", 
               func_tsk = func_tsk)
    
    
@author: kmi
"""
import numpy as np
import pandas as pd
from itertools import chain

##### Function defuzzifier #########################
'''
This function is used for defuzifying the input of rules 
by using TSK or Mamdani approach
    
    data -- 
    rule --   defines the list of rules 
    range_output --  maximum and minimum values of the data 
    names_varoutput --  names of the output variables
    varout_mf --   parameters of membership functions on the output variable
    miu_rule - matreix containing the values of mu 
    type_defuz -- type of defuzzifier
    type_model = "MAMDANI" or "TSK"
    func_tsk  - random uniformly distributed cluster centers or 
                define the linear equations on the consequent parts 
                of the fuzzy IF-THEN rules.

'''

def defuzzifier(data , rule, range_output, names_varoutput, 
   varout_mf, miu_rule, type_defuz, type_model, 
   func_tsk):
    
#   defuz = np.zeros([len(data),1]) 
#   def_temp = np.zeros([len(data),1])
#    ################# Code for Mamdani ######################################
#   if (type_model == 1 or type_model == "MAMDANI") :        
#       #rule_temp = np.append(rule_temp, list(chain(*rule)))
#      if (names_varoutput is None):
#          print("Please define the names of the output variable.")
#        
#      if (varout_mf is None) :
#          print("Please define the parameters of membership functions on the output variable.")       
#          cum = np.zeros(len(data),np.shape(varout.mf)[1])
#          div = np.zeros(len(data),np.shape(varout.mf)[1])

   ###################### Code for TSK     ###################################   
   if(type_model == 2 or type_model == "TSK") : 
       
       defuz = np.zeros([len(data),1]) 
       
#       if (any(func_tsk) == None):
#          print("Please define the linear equations on the consequent parts of the fuzzy IF-THEN rules.")
#            
       for k in range(len(data)): 
          
          data_m = [data[k]]
          data_m = np.asarray(data_m)
          if (np.shape(func_tsk)[1] > 1): 
             func_tsk_var = func_tsk[:, :-1] 
             func_tsk_cont = func_tsk[:,[np.shape(func_tsk)[1]-1]]
             ff = np.matmul(func_tsk_var, data_m.T) + func_tsk_cont
          elif (np.shape(func_tsk)[1] == 1) :
             ff = func_tsk
    
          miu_rule_t = np.asanyarray([miu_rule[k]])
          cum = np.matmul(miu_rule_t,ff)
          div = [np.sum(miu_rule_t)]
          
          defuz[k]= np.true_divide([cum],[div])   ####
#                    
          if (div == 0): 
             defuz[k] = 0
          else:
             defuz[k] = [np.divide(cum,div)]
             if (defuz[k] > np.max(range_output)) :
                defuz[k] = [np.max(range_output)]
             elif (defuz[k] < min(range_output)): 
                defuz[k] = [np.min(range_output)]
       return(defuz)
"""
@author: Manish Kakar
Created on Mon Jun 17 12:33:06 2019
 The Gas Furnance dataset is taken from Box and Jenkins. 
 It consists of 292 consecutive 
 values of methane at time (t - 4), and the CO2 produced in a furnance at time (t - 1) as input 
 variables, with the produced CO2 at time (t) as an output variable. So, each training data 
 point consists of [u(t - 4), y(t - 1), y(t)], where u is methane and y is CO2.

@Reference:

G. E. P. Box and G. M. Jenkins, "Time series analysis, forecasting and control", San Fransisco, CA: Holden Day (1970).

N.K. Kasabov and Q. Song, "DENFIS: Dynamic evolving neural-fuzzy inference system and its Application for time-series prediction", 
 IEEE Transactions on Fuzzy Systems, vol. 10, no. 2, pp. 144 - 154 (2002).

@Example :
import pandas as pd    
dataframe = pd.read_csv('GasFurnace205pts.csv', usecols = [1,2,3] )
ts = dataframe.values
cc = ECM_CLUST(ts = ts, Dthr = 0.2)
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance
import numpy.linalg


mackey =  './data/Mackey.csv'
gasf = './data/GasFurnace292pts.csv'

if mackey:
    dataframe = pd.read_csv(mackey)
else:
    dataframe = pd.read_csv(gasf, usecols = [1,2,3])
    
### Change the file Name in accordance to the data type in the example file 

#dataframe = pd.read_csv('./data/GasFurnace205pts.csv', header = 0, usecols = [1,2,3] )
#dataframe = pd.read_csv('./data/Mackey.csv')
ts = dataframe.values

#Create first cluster
def ECM_CLUST(ts, Dthr):
    Cc1 = ts[0:1]
    Ccj =[]
    Ccj = Cc1 
    Ruj = []
    Ru1 = 0
    Ruj.append(Ru1)
    nrows = np.shape(ts)[0]
    ncols = np.shape(ts)[1]
    Norm_Euc_Dist = []
    Dij = []
    Sij = []
    #Dthr = 0.2
    Rua = 0
    Sia = 0
    new_cls = []
    for datapoints in range(1,nrows):
        new_list =np.array([])
        for centers in range(len(Ccj)):   #np.shape(Ccj)[0]):       
            Norm_Euc_Dist = distance.euclidean(ts[datapoints], Ccj[centers])
            Norm_Euc_Dist = [Norm_Euc_Dist]/np.sqrt(ncols)
            new_list = np.append(new_list,[Norm_Euc_Dist])
        Dij = new_list
        #print("\n\nvalue of Dij is \n  {}".format(Dij))
            
        indx = np.argmin((Dij))  
        if any(Dij[indx] <= Ruj):
           pass
            #print("In cluster Cm")
        else:           
            Sij = Dij  + Ruj
            Sij.tolist()
            Sij = Sij.transpose()
            ind_s = np.argmin(Sij)
            
#        if ( Sij[ind_s] > 2*Dthr):
#            Ccj = np.vstack((Ccj, ts[datapoints]))
#            Ruj = np.append(Ruj,0)    
#        else:    
            Sia = np.min(Sij[ind_s])
            Rua = 0.5*Sia
            
            if (Sia > 2*Dthr) :
               #Make a new cluster
               Ccj = np.vstack((Ccj, ts[datapoints]))
               Ruj = np.append(Ruj,0)
            else:         
               temp = ts[datapoints] - Ccj[ind_s]
               d_temp = np.sqrt(np.sum(temp**2))
               ratio = np.abs(d_temp - Rua)/d_temp
               new_vec = np.multiply(ratio, temp)
               new_cls = np.add(Ccj[ind_s], new_vec)
               Ccj[ind_s] = new_cls
               Ruj[ind_s ] =  Rua
    res  = np.vstack(Ccj)
    return res
#print("\n\nvalue of centers is \n  {}".format(res))

           

 
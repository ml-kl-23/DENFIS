# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:35:42 2019

PS ! CHOOSE THE CORRECT DATATYPE ( MACKEY or GASFURNACE) IN THE ECM.py FILE


The task of the DENFIS model is to provide an identification for the CO2 concentration 
y(t) given the methane gas portion from four time steps before u(t−4) and the 
last CO2 concentration y(t−1).

The Gas Furnance dataset is taken from Box and Jenkins. 
 It consists of 292 consecutive 
 values of methane at time (t - 4), and the CO2 produced in a furnance at time (t - 1) as input 
 variables, with the produced CO2 at time (t) as an output variable. So, each training data 
 point consists of [u(t - 4), y(t - 1), y(t)], where u is methane and y is CO2.


@Reference:

G. E. P. Box and G. M. Jenkins, "Time series analysis, forecasting and control", San Fransisco, CA: Holden Day (1970).

N.K. Kasabov and Q. Song, "DENFIS: Dynamic evolving neural-fuzzy inference system and its Application for time-series prediction", 
 IEEE Transactions on Fuzzy Systems, vol. 10, no. 2, pp. 144 - 154 (2002).

@author: MANISH KAKAR



"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from DENFIS2 import DENFIS
from denfis_predict2 import denfis_predict

def add_one(number):
    return number + 1


gasf = True

np.random.seed(101)

dataframe2 = pd.read_csv('./data/GasFurnace292pts.csv', usecols = [1,2,3])
ts2 = dataframe2.values
ts = np.array(ts2)

## decides the ratio for training and testing data
ratio = 0.75
data_shape = np.shape(ts)

training_data =ts[0:round(ratio*len(ts))] # training data used 70% for testing
fitting_data = np.delete(ts[0:round(ratio*len(ts))],-1,1)
testing_data =np.delete(ts[round(ratio*len(ts)):len(ts)],-1,1 )
real_val_test = ts[round(ratio*len(ts)):len(ts), np.shape(ts)[1] -1]

training_data2 =training_data.copy()
fitting_data2 =fitting_data.copy() 
testing_data2 =testing_data.copy()
real_val_test2 = real_val_test.copy()


min_range = np.min(ts,axis=0)
max_range = np.max(ts,axis=0)
range_data = np.asarray([min_range.T, max_range.T])  
range_data2 = range_data.copy()

### Create the model
mod = DENFIS(data_train = training_data2, range_data = range_data2, Dthr = 0.1, max_iter = 300, step_size = 0.01, d =2 )

## Fit the model
result_fitting = denfis_predict(ModelInput = mod, NewData= fitting_data2)

## Plot the fitting data
plt.subplot(2,1,1)
plt.plot(result_fitting)
plt.plot(training_data2[:,-1])
plt.xlabel("#Data Points")
plt.ylabel("CO2")
plt.legend(("Predicted", "True"))
plt.title("Fitting Phase")
plt.show()

## Predict and compare
result_testing = denfis_predict(ModelInput = mod, NewData= testing_data2)

## error calculation
y_pred = result_testing
y_real = np.asarray([real_val_test2])
y_real = y_real.T
benchmarking = np.hstack([y_pred,y_real])

#RMSE =
residuals = (y_real - y_pred)
### Replace NAN with 0
ifNan = np.isnan(residuals)
residuals[ifNan]  = 0

#### Replace NaN with interpolated values
#nans, a= np.isnan(residuals), lambda z: z.nonzero()[0]
#residuals[nans]= np.interp(a(nans), a(~nans), residuals[~nans])

MSE = np.mean(np.power(residuals,2))
RMSE = np.sqrt(MSE)
print("RMSE Value is {}".format(RMSE))
print("MSE Value is {}".format(MSE))

#### Plot 
plt.subplot(2,1,2)
plt.plot(y_pred)
plt.plot(y_real)
plt.legend(("Predicted", "True"))
plt.xlabel("#Data Points")
plt.ylabel("CO2")
plt.title("Testing Phase")
plt.show()


gasf = False
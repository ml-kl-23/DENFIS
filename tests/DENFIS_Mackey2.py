# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:27:29 2019

The Mackey-Glass chaotic time series is defined by the following delayed differential equation:
 
 d_x(t) / d_t = (a * x(t - \tau) / (1 + x(t - \tau) ^ 10)) - b * x(t)}
 
For this dataset, we generated 1000 samples, with input parameters as follows:
    
  a = 0.2
  b = 0.1
  tau = 17
  x_0 = 1.2
  d_t = 1

The dataset is embedded in the following way: 
input variables: x(t - 18), x(t - 12), x(t - 6)}, x(t)
output variable: x(t + 6)

@Reference:

M. Mackey and L. Glass, "Oscillation and chaos in physiological control systems", Science, vol. 197, pp. 287 - 289 (1977).

N.K. Kasabov and Q. Song, "DENFIS: Dynamic evolving neural-fuzzy inference system and its Application for time-series prediction", 
 IEEE Transactions on Fuzzy Systems, vol. 10, no. 2, pp. 144 - 154 (2002).

@author: Manish Kakar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DENFIS2 import DENFIS
from denfis_predict2 import denfis_predict

mackey=True

def add_one(number):
    return number + 1

np.random.seed(101)


dataframe2 = pd.read_csv('./data/Mackey.csv' )
ts = dataframe2.values
ts1 = np.array(ts)

## decides the ratio for training and testing data
ratio = 0.75

training_data = ts1[0:int(round(len(ts1))*ratio),:]
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


mod = DENFIS(data_train = training_data2, range_data = range_data, Dthr = 0.1, max_iter = 300, step_size = 0.01, d =2 )


## Fit the model
result_fitting = denfis_predict(ModelInput = mod, NewData= fitting_data2)

## Plot the fitting data
plt.subplot(2,1,1)
plt.plot(result_fitting)
plt.plot(training_data[:,-1])
plt.xlabel("#Data Points")
plt.ylabel("Amplitude ")
plt.legend(("Predicted", "True"))
plt.title("Fitting Phase")

#
## Predict and compare
result_testing = denfis_predict(ModelInput = mod, NewData= testing_data2)


## Error 
## error calculation
y_pred = result_testing
y_real = np.asarray([real_val_test2])
y_real = y_real.T
benchmarking = np.hstack([y_pred,y_real])
#RMSE =
#RMSE =
residuals = (y_real - y_pred)
### Replace NAN with 0
ifNan = np.isnan(residuals)
residuals[ifNan]  = 0

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
plt.ylabel("Amp")
plt.title("Testing Phase")
plt.show()


mackey=False
from  tslib.src.models.TSModel import TSmodel
import time
from tslib.src.hdf_util import read_data
import numpy as np
import time
import pandas as pd
from tslib.src import tsUtils

# load data, means and obsevations
Data = read_data('tests/testdata/MixtureTS2.h5')
trainData = Data['obs']
means = Data['means']

# Specify Parameters 
"""
L : 		Number of rows in each submodel
T0: 		the minimum number of data points to fit the model (i.e. if less the model will not be fitted)
gamma: 		the ratio of new data points after which the sub-model will be reconstructed 
k: 			the number of singular values to keep
rectFactor: the ration of no. columns to the number of rows in each sub-model
"""
L = 100
T0 = 1000
gamma = 0.5
k = 3
rectFactor = 10

# determine the number of points to be predicted 
noPredictions = 1000
TrainIndex = len(trainData) - noPredictions

#init and fit model
a = TSmodel(k, L, gamma, T0, rectFactor = rectFactor)
t = time.time()
a.UpdateModel(trainData[:TrainIndex])

# print time to fit model
print (time.time() - t), (time.time() - t) / a.MUpdateIndex

# impute and calculate imputation error 
t = a.denoiseTS()
print 'error of imputation:', np.sqrt(np.mean((t - trainData[:len(t)]) ** 2))
RMSE1 = tsUtils.rmse(trainData[:len(t)], t)
RMSE2 = tsUtils.rmse(means[:len(t)], t)
print 'RMSE of imputation vs obs:', RMSE1
print 'RMSE of imputation vs means:', RMSE2

# predict and calculate predictions error 
predictionsindices = np.arange(TrainIndex, len(trainData))
predictionsindices = predictionsindices[:noPredictions]
predictions = [a.predict(dataPoints=trainData[i - a.L + 1:i], NoModels=10) for i in predictionsindices]
RMSE1 = tsUtils.rmse(trainData[TrainIndex: len(trainData)], predictions)
RMSE2 = tsUtils.rmse(means[TrainIndex: len(trainData)], predictions)
print 'RMSE of forecasting vs obs:', RMSE1
print 'RMSE of forecasting vs means:', RMSE2

print len(a.models), a.models[0].N, a.models[0].M
t = time.time()

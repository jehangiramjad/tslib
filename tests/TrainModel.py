from  tslib.src.models.TSModel import TSmodel
import time
from tslib.src.hdf_util import read_data
import numpy as np
import time
import pandas as pd
from tslib.src import tsUtils

# load data
Data = read_data('C:\Users\Abdul\Dropbox (MIT)\Time Series Project\\tslib2\\tests\\testdata\MixtureTS.h5')
trainData = Data['obs']
means = Data['means']

T = 100
T0 = 1000
gamma = 0.5
k = 3
Train = 800000
d = T ** 2 * 10
j = 0
noPredictions = 1000
TrainIndex = len(trainData) - noPredictions

a = TSmodel(k, T, gamma, T0, NoRows=True)
t = time.time()
a.UpdateModel(trainData[:TrainIndex])

print (time.time() - t), (time.time() - t) / a.MUpdateIndex
t = a.denoiseTS()
print 'error of imputation:', np.sqrt(np.mean((t - trainData[:len(t)]) ** 2))
RMSE1 = tsUtils.rmse(trainData[:len(t)], t)
RMSE2 = tsUtils.rmse(means[:len(t)], t)
print 'RMSE of imputation vs obs:', RMSE1
print 'RMSE of imputation vs means:', RMSE2

predictionsindices = np.arange(TrainIndex, len(trainData))
predictionsindices = predictionsindices[:noPredictions]

predictions = [a.predict(dataPoints=trainData[i - a.L + 1:i], NoModels=10) for i in predictionsindices]

RMSE1 = tsUtils.rmse(trainData[TrainIndex: len(trainData)], predictions)
RMSE2 = tsUtils.rmse(means[TrainIndex: len(trainData)], predictions)
print 'RMSE of forecasting vs obs:', RMSE1
print 'RMSE of forecasting vs means:', RMSE2

print len(a.models), a.models[0].N, a.models[0].M
t = time.time()

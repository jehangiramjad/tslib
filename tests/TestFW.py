import sys

from  tslib.src.models.TSModel import TSmodel
from time import time as clock
from tslib.src.data import generateHarmonics as gH
from  tslib.src.data import generateTrend as gT
import tslib.src.data.generateARMA as gA
import numpy as np
import tslib.src.tsUtils as tsUtils
import copy
import pandas as pd
from tslib.src.hdf_util import read_data


def armaDataTest(timeSteps):
    arLags = []
    maLags = []

    startingArray = np.zeros(np.max([len(arLags), len(maLags)]))  # start with all 0's
    noiseMean = 0.0
    noiseSD = 1.0

    (observedArray, meanArray, errorArray) = gA.generate(arLags, maLags, startingArray, timeSteps, noiseMean, noiseSD)

    return (observedArray, meanArray)


def trendDataTest(timeSteps):
    dampening = 2.0 * float(1.0 / timeSteps)
    power = 0.35
    displacement = -2.5

    f1 = gT.linearTrendFn
    data = gT.generate(f1, power=power, displacement=displacement, timeSteps=timeSteps)

    f2 = gT.logTrendFn
    data += gT.generate(f2, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

    f3 = gT.negExpTrendFn
    t3 = gT.generate(f3, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

    # plt.plot(t2)
    # plt.show()

    return data


def harmonicDataTest(timeSteps):
    sineCoeffs = [-2.0, 3.0]
    sinePeriods = [26.0, 30.0]

    cosineCoeffs = [-2.5]
    cosinePeriods = [16.0]

    data = gH.generate(sineCoeffs, sinePeriods, cosineCoeffs, cosinePeriods, timeSteps)
    # plt.plot(data)
    # plt.show()

    return data


timeSteps = 2 * 10 ** 6 + 1

# # start TS model Testing

harmonicsTS = harmonicDataTest(timeSteps)
trendTS = trendDataTest(timeSteps)
(armaTS, armaMeanTS) = armaDataTest(timeSteps)

means = harmonicsTS + trendTS + armaMeanTS
observedArray = harmonicsTS + trendTS + armaTS
max1 = np.nanmax(observedArray)
min1 = np.nanmin(observedArray)
max2 = np.nanmax(means)
min2 = np.nanmin(means)
max = np.max([max1, max2])
min = np.min([min1, min2])
#
# observedArray = tsUtils.normalize(observedArray, max, min)
# means = tsUtils.normalize(means, max, min)


# load data
# Data = read_data('testdata/MixtureTS.h5')
# observedArray = Data['obs']
# means = Data['means']
p = 0.9
(trainDataO, pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(observedArray[:]), p)
noPredictions = 1000
TrainIndex = len(trainDataO) - noPredictions
MS = 200
ga = 0.5
ModelSize = [i for i in [100]]
gammas = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
RMSE = np.zeros([len(ModelSize), 4])
Timing = np.zeros([len(ModelSize)])
k = 3

for ii, MS in enumerate(ModelSize):

    # a = TSmodel(k, MS, ga, MS ** 2, NoRows=True)
    # T = 1000000 + MS ** 2 * 10 - 1
    # d = int(MS)
    # j = 0
    # predictionIndices = np.arange(MS ** 2 * 10, T, d) + d

    # select = np.random.random(len(predictionIndices)) < 0.1
    # predictionIndices = predictionIndices[select]

    # predictions = np.zeros([predictionIndices.shape[0],2])
    trainData = copy.copy(trainDataO)
    a = TSmodel(k, MS, ga, 1000, NoRows=True)
    t = clock()
    a.UpdateModel(trainData[:TrainIndex], patch=True)

    Timing[ii] = (clock() - t)
    print '--------------------------------'
    print 'total time:', (clock() - t)
    Timing[ii] = Timing[ii] / (a.MUpdateIndex)
    print MS, ga, 'time:    ', Timing[ii], '\n'

    tsDenoised = a.denoiseTS()
    print 'ModelInfo:', len(a.models), a.models[0].M, a.models[0].N
    RMSE[ii, 0] = tsUtils.rmse(observedArray[:len(tsDenoised)], tsDenoised)
    RMSE[ii, 1] = tsUtils.rmse(means[:len(tsDenoised)], tsDenoised)
    print 'RMSE of imputation vs obs:', RMSE[ii, 0]
    print 'RMSE of imputation vs means:', RMSE[ii, 1]

    predictionsindices = np.arange(TrainIndex, len(trainData))
    predictionsindices = predictionsindices[:noPredictions]

    predictions = [a.predict(dataPoints=observedArray[i - a.L + 1:i], NoModels=10) for i in predictionsindices]

    RMSE[ii, 2] = tsUtils.rmse(observedArray[TrainIndex: len(trainData)], predictions)
    RMSE[ii, 3] = tsUtils.rmse(means[TrainIndex: len(trainData)], predictions)

    print 'RMSE of forecasting vs obs:', RMSE[ii, 2]
    print 'RMSE of forecasting vs means:', RMSE[ii, 3]
    print len(tsDenoised), a.MUpdateIndex
    print '-----------------------------------'

    # a.updateTS(trainData[0:MS ** 2 * 10])
    # a.fitModels()
    # print(a.ReconIndex, MS ** 2 * 10, a.TimeSeriesIndex)

    # for w,i in enumerate(range(MS ** 2 * 10, T, d)):


    #     t = clock()
    #     a.updateTS(trainData[i:i + d])
    #     a.fitModels()

    #     Timing[ii, q] += (clock() - t)
    #     if select[w]:
    #         Reconstructed = a.denoiseTS()
    #         length = len(Reconstructed)
    #         RMSE[ii, 0, q] += tsUtils.rmse(meanTS[:length], Reconstructed)
    #         RMSE[ii, 1, q] += tsUtils.rmse(combinedTS[:length], Reconstructed)
    #         predictions[j,0] = a.predict()
    #         predictions[j,1] = a.predict(NoModels = 10)
    #         j += 1




    # RMSE[ii, 0, q] /= j
    # RMSE[ii, 1, q] /= j
    # RMSE[ii, 2, q] = tsUtils.rmse(meanTS[predictionIndices], predictions[:,0])
    # RMSE[ii, 3, q] = tsUtils.rmse(combinedTS[predictionIndices], predictions[:,0])
    # RMSE[ii, 4, q] = tsUtils.rmse(meanTS[predictionIndices], predictions[:,1])
    # RMSE[ii, 5, q] = tsUtils.rmse(combinedTS[predictionIndices], predictions[:,1])
    # print(" \n RMSE (prediction vs means)  = %f" % RMSE[ii, 2, q])
    # print(" \n RMSE (prediction vs obs)  = %f" % RMSE[ii, 3, q])

# DF = pd.DataFrame(RMSE, columns=['ImpErrMean', 'ImpErrObs', 'ForErrObs', 'ForErrMean'])
# DF['timing'] = Timing
# DF['ModelSize'] = ModelSize
# DF.to_csv('Results_ht_vs_ModelSize_patch_billion.csv')

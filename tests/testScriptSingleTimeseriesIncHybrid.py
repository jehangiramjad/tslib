#############################################################
#
# Single-Dimensional Time Series Imputation and Forecasting
#
# You need to ensure that this script is called from
# the tslib/ parent directory or tslib/tests/ directory:
#
# 1. python tests/testScriptSingleTimeseries.py
# 2. python testScriptSingleTimeseries.py
#
#############################################################
import sys, os
from time import clock
sys.path.append("../..")
sys.path.append("C:\\Users\Abdul\Dropbox (MIT)\Time Series Project\\tslib2\\tslib")
sys.path.append("..")
sys.path.append(os.getcwd())

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy

from tslib.src.data import generateHarmonics as gH
from  tslib.src.data import generateTrend as gT
import tslib.src.data.generateARMA as gA
from  tslib.src.models.tsSVDModel import SVDModel

import tslib.src.tsUtils as tsUtils


def armaDataTest(timeSteps):

    arLags = [0.4, 0.3, 0.2]
    maLags = [0.5, 0.1]

    startingArray = np.zeros(np.max([len(arLags), len(maLags)])) # start with all 0's
    noiseMean = 0.0
    noiseSD = 1.0

    (observedArray, meanArray, errorArray) = gA.generate(arLags, maLags, startingArray, timeSteps, noiseMean, noiseSD)

    return (observedArray, meanArray)

def trendDataTest(timeSteps):

    dampening = 2.0*float(1.0/timeSteps)
    power = 0.35
    displacement = -2.5

    f1 = gT.linearTrendFn
    data = gT.generate(f1, power=power, displacement=displacement, timeSteps=timeSteps)

    f2 = gT.logTrendFn
    data += gT.generate(f2, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

    f3 = gT.negExpTrendFn
    t3 = gT.generate(f3, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

    #plt.plot(t2)
    #plt.show()

    return data


def harmonicDataTest(timeSteps):

    sineCoeffs = [-2.0, 3.0]
    sinePeriods = [26.0, 30.0]

    cosineCoeffs = [-2.5]
    cosinePeriods = [16.0]

    data = gH.generate(sineCoeffs, sinePeriods, cosineCoeffs, cosinePeriods, timeSteps)
    #plt.plot(data)
    #plt.show()

    return data




# test for a single time series imputation and forecasting
def testSingleTS():

    print("------------------- Test # 1 (Single TS). ------------------------")
    p = 0.8
    N = 100
    M = 100
    NC = 100
    timeSteps = N*(M+NC)

    # train/test split
    trainProp = 1
    M1 = int(trainProp * (M+NC))

    trainPoints = N*M1
    Ite = 10

    d = 1
    RMSE = np.zeros([NC/d, 6, Ite])
    timing = np.zeros([NC / d, 3, Ite])
    for it in range(Ite):
        print (it)
        # print("Generating data...")
        harmonicsTS = harmonicDataTest(timeSteps)
        trendTS = trendDataTest(timeSteps)
        (armaTS, armaMeanTS) = armaDataTest(timeSteps)

        meanTS = harmonicsTS + trendTS + armaMeanTS
        combinedTS = harmonicsTS + trendTS + armaTS

        #normalize the values to all lie within [-1, 1] -- helps with RMSE comparisons
        # can use the tsUtils.unnormalize() function to convert everything back to the original range at the end, if needed
        max1 = np.nanmax(combinedTS)
        min1 = np.nanmin(combinedTS)
        max2 = np.nanmax(meanTS)
        min2 = np.nanmin(meanTS)
        max = np.max([max1, max2])
        min = np.min([min1, min2])

        combinedTS = tsUtils.normalize(combinedTS, max, min)
        meanTS = tsUtils.normalize(meanTS, max, min)

        # produce timestamps
        timestamps = np.arange('2017-09-10 20:30:00', timeSteps, dtype='datetime64[1m]') # arbitrary start date

        # split the data
        trainDataMaster = combinedTS[0:trainPoints] # need this as the true realized values for comparisons later
        meanTrainData = meanTS[0:trainPoints] # this is only needed for various statistical comparisons later

        # randomly hide training data: choose between randomly hiding entries or randomly hiding consecutive entries
        (trainData, pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(trainDataMaster), p)

        # now further hide consecutive entries for a very small fraction of entries in the eventual training matrix
        (trainData, pObservation) = tsUtils.randomlyHideConsecutiveEntries(copy.deepcopy(trainData), 1, int(M1 * 0.25), M1)

        # interpolating Nans with linear interpolation
        #trainData = tsUtils.nanInterpolateHelper(trainData)

        # test data and hidden truth

        # time stamps
        trainTimestamps = timestamps[0:trainPoints]

        # once we have interpolated, pObservation should be set back to 1.0
        pObservation = 1.0

        # create pandas df
        key1 = 't1'
        trainMasterDF = pd.DataFrame(index=trainTimestamps[:N*M], data={key1: trainDataMaster[:N*M]}) # needed for reference later
        trainDF = pd.DataFrame(index=trainTimestamps[:N*M], data={key1: trainData[:N*M]})
        meanTrainDF = pd.DataFrame(index=trainTimestamps[:N*M], data={key1: meanTrainData[:N*M]})

        # train the model
        # print("Training the model (imputing)...")
        # print('SVD')
        nbrSingValuesToKeep = 5
        mod = SVDModel(key1, nbrSingValuesToKeep, N, M, probObservation=pObservation, svdMethod='numpy', otherSeriesKeysArray=[], includePastDataOnly=True)
        mod.fit(trainDF)
        imputedDf = mod.denoisedDF()

        # print(" RMSE (training imputation vs mean) = %f" %tsUtils.rmse(meanTrainDF[key1].values, imputedDf[key1].values))
        # print(" RMSE (training imputation vs obs)  = %f" %tsUtils.rmse(trainMasterDF[key1].values, imputedDf[key1].values))
        PM1 = int(0.1*M)
        PM2 = int(0.025 * M)

        M1 = M

        mod2 = copy.deepcopy(mod)
        mod3 = copy.deepcopy(mod)
        D1 = np.ones([N,1])
        D2 = np.ones([N, 1])
        for i in range(NC/d):


            M1 = M1+d
            D = trainData[N*(M1-d):N*M1]

            D1 = np.concatenate((D1,D.reshape(-1,1)),1)
            D2 = np.concatenate((D2,D.reshape(-1,1)),1)
            if i % PM1 !=0:
                t = clock()
                imputedNew = mod.denoisedDFNew(D)
                timing[i,0,it] = clock()-t
            else:
                t = clock()

                imputedNew = mod.denoisedDFNew(D1[:,1:].flatten(),'UP')
                D1 = D.reshape(-1,1)

                timing[i, 0, it] = clock() - t

            if i % PM2 !=0:
                t = clock()
                imputedNew2 = mod2.denoisedDFNew(D)
                timing[i,1,it] = clock()-t
            else:
                t = clock()
                imputedNew2 = mod2.denoisedDFNew(D2[:,1:].flatten(),'UP')
                D2 = D.reshape(-1,1)
                timing[i, 1, it] = clock() - t

            t = clock()
            imputedNew3 = mod3.denoisedDFNew(D, 'UP')
            timing[i, 2, it] = clock() - t

            RMSE[i,0,it] = tsUtils.rmse(imputedNew[-N:],meanTrainData[N*(M1-d):N*M1] )
            RMSE[i, 1,it] = tsUtils.rmse(imputedNew[-N:],trainDataMaster[N*(M1-d):N*M1])
            RMSE[i,2,it] = tsUtils.rmse(imputedNew2[-N:],meanTrainData[N*(M1-d):N*M1] )
            RMSE[i, 3,it] = tsUtils.rmse(imputedNew2[-N:],trainDataMaster[N*(M1-d):N*M1])
            RMSE[i, 4, it] = tsUtils.rmse(imputedNew3, meanTrainData[N * (M1 - d):N * M1])
            RMSE[i, 5, it] = tsUtils.rmse(imputedNew3, trainDataMaster[N * (M1 - d):N * M1])

            # print(i," RMSE (training imputation vs mean) = %f" %tsUtils.rmse(imputedNew,meanTrainData[N*(M1-d):N*M1] ))
            # print(i," RMSE (training imputation vs obs)  = %f" %tsUtils.rmse(imputedNew,trainDataMaster[N*(M1-d):N*M1] ))
    # print("Plotting...")
    # plt.figure()
    # plt.plot(np.arange(d,NC+d,d),np.mean(RMSE[:, 0,:],1), label= 'folding-in vs. mean')
    # plt.plot(np.arange(d,NC+d,d),np.mean(RMSE[:, 1,:],1),'--',label= 'folding-in vs. obs')
    # plt.plot(np.arange(d,NC+d,d),np.mean(RMSE[:, 2,:],1), label= 'Updating vs. mean')
    # plt.plot(np.arange(d,NC+d,d),np.mean(RMSE[:, 3,:],1),'--',label= 'Updating vs. obs')
    # plt.xlabel('Columns added')
    # plt.ylabel('RMSE (imputation) ')
    # plt.legend()
    # print(np.mean(timing[:,0,:]),np.mean(timing[:,1,:]))
    # plt.figure()
    # plt.semilogy(np.arange(d,NC+d,d),np.mean(timing[:, 0,:],1), label= 'folding-in time')
    # plt.semilogy(np.arange(d,NC+d,d),np.mean(timing[:, 1,:],1),'--',label= 'Updating time')
    # plt.legend()
    # plt.xlabel('Columns added')
    # plt.ylabel('time (seconds)')
    # plt.show()
    return RMSE, timing


def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    rmse, timing = testSingleTS()
    return rmse, timing
    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":
    d =1
    q = 10
    rmse, timing = main()
    NC = rmse.shape[0]*d
    plt.figure()
    plt.plot(np.arange(d, NC + d, d)[::q], np.mean(rmse[::q, 0, :], 1), label='Hybrid (0.1) vs. mean')
    plt.plot(np.arange(d, NC + d, d)[::q], np.mean(rmse[::q, 1, :], 1), '--', label='Hybrid (0.1) vs. obs')
    plt.plot(np.arange(d, NC + d, d)[::q], np.mean(rmse[::q, 2, :], 1), label='Hybrid (0.025) vs. mean')
    plt.plot(np.arange(d, NC + d, d)[::q], np.mean(rmse[::q, 3, :], 1), '--', label='Hybrid (0.025) vs. obs')
    plt.plot(np.arange(d, NC + d, d)[::q], np.mean(rmse[::q, 4, :], 1), label='Updating vs. mean')
    plt.plot(np.arange(d, NC + d, d)[::q], np.mean(rmse[::q, 5, :], 1), '--', label='Updating vs. obs')


    plt.xlabel('Columns added')
    plt.ylabel('RMSE (imputation) ')
    plt.legend()
    print(np.sum(np.mean(timing[:, 0,:],1)),np.sum(np.mean(timing[:, 1,:],1)),np.sum(np.mean(timing[:, 2,:],1)))
    plt.figure()
    plt.semilogy(np.arange(d,NC+d,d)[::q],np.mean(timing[::q, 0,:],1), label= 'Hybrid time')
    plt.semilogy(np.arange(d,NC+d,d)[::q],np.mean(timing[::q, 1,:],1),'--',label= 'Updating time')

    plt.legend()
    plt.xlabel('Columns added')
    plt.ylabel('time (seconds)')
    plt.show()
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
    p = 1
    N = 1000
    M = 1000
    NC = 100
    timeSteps = N*(M+NC)

    # train/test split
    trainProp = 1
    M1 = int(trainProp * (M+NC))

    trainPoints = N*M1
    Ite = 5

    d = 1
    RMSE = np.zeros([NC/d, 6, Ite])
    timing = np.zeros([NC / d, 3, Ite])
    prediction = np.zeros([NC / d, 5, Ite])
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

        M1 = M

        Means =[]
        Obs =[]

        mod2 = copy.deepcopy(mod)
        mod3 = copy.deepcopy(mod)
        for i in range(NC/d - 1):
            print(i)
            M1 = M1+d
            D = trainData[N*(M1-d):N*M1]
            t = clock()
            mod.updateSVD(D)
            imputedNew = mod.denoisedTS([N*(M1-d),N*M1])
            timing[i,0,it] = clock()-t
            pastPoints = trainData[N*(M1-1) -1 :N*M1]
            keyToSeriesDFNew = pd.DataFrame(data={key1: pastPoints})
            prediction[i, 0, it] = mod.predict(pd.DataFrame(data={}), keyToSeriesDFNew)
            prediction[i, 4, it] = trainData[N*M1]
            prediction[i, 3, it] = meanTS[N*M1]

            t = clock()
            mod2.updateSVD(D,'UP')
            imputedNew2 = mod2.denoisedTS([N*(M1-d),N*M1])
            timing[i, 1, it] = clock() - t
            prediction[i, 1, it] = mod2.predict(pd.DataFrame(data={}), keyToSeriesDFNew)
            t = clock()
            # if it == 0:
            #      imputedNew3 = mod3.denoisedDFNew(D,'Full')
            #      timing[i, 2, it] = clock() - t
            #      RMSE[i, 4, it] = tsUtils.rmse(imputedNew3, meanTrainData[N * (M1 - d):N * M1])
            #      RMSE[i, 5, it] = tsUtils.rmse(imputedNew3, trainDataMaster[N * (M1 - d):N * M1])
            # prediction[i,2,it] = mod3.predict(pd.DataFrame(data={}), keyToSeriesDFNew)

            RMSE[i,0,it] = tsUtils.rmse(imputedNew,meanTrainData[N*(M1-d):N*M1] )
            RMSE[i, 1,it] = tsUtils.rmse(imputedNew,trainDataMaster[N*(M1-d):N*M1])
            RMSE[i,2,it] = tsUtils.rmse(imputedNew2,meanTrainData[N*(M1-d):N*M1] )
            RMSE[i, 3,it] = tsUtils.rmse(imputedNew2,trainDataMaster[N*(M1-d):N*M1])
    Means = prediction[:, 3, 0]
    Obs  = prediction[:, 4, 0]
    print(i," RMSE folding-in (Prediction vs mean) = %f" %tsUtils.rmse(np.mean(prediction[:, 0, :],1),Means))
    print(i," RMSE folding-in (Prediction vs obs)  = %f" %tsUtils.rmse(np.mean(prediction[:, 0, :],1),Obs ))
    print(i," RMSE Updating (Prediction vs mean) = %f" %tsUtils.rmse(np.mean(prediction[:, 1, :],1),Means))
    print(i," RMSE Updating (Prediction vs obs)  = %f" %tsUtils.rmse(np.mean(prediction[:, 1, :],1),Obs ))
    # print(i," RMSE Recons (Prediction vs mean) = %f" %tsUtils.rmse(np.mean(prediction[:, 2, :],1),Means))
    # print(i," RMSE Recons (Prediction vs obs)  = %f" %tsUtils.rmse(np.mean(prediction[:, 2, :],1),Obs ))

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
    return RMSE, timing, prediction


def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    rmse, timing, prediction = testSingleTS()
    return rmse, timing, prediction
    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":
    d =1
    q = 1
    rmse, timing ,prediction= main()
    NC = rmse.shape[0]*d
    plt.figure()
    plt.plot(np.arange(d, NC + d, d)[::q], np.mean(rmse[::q, 0, :], 1), label='folding-in vs. mean')
    plt.plot(np.arange(d, NC + d, d)[::q], np.mean(rmse[::q, 1, :], 1), '--', label='folding-in vs. obs')
    plt.plot(np.arange(d, NC + d, d)[::q], np.mean(rmse[::q, 2, :], 1), label='Updating vs. mean')
    plt.plot(np.arange(d, NC + d, d)[::q], np.mean(rmse[::q, 3, :], 1), '--', label='Updating vs. obs')
    # plt.plot(np.arange(d, NC + d, d)[::q], rmse[::q, 4, 0], label='Recomputing SVD vs. mean')
    # plt.plot(np.arange(d, NC + d, d)[::q], rmse[::q, 5, 0], '--', label='Recomputing SVD vs. obs')

    plt.xlabel('Columns added')
    plt.ylabel('RMSE (imputation) ')
    plt.legend()

    plt.figure()
    plt.semilogy(np.arange(d,NC+d,d)[::q],np.mean(timing[::q, 0,:],1), label= 'folding-in time')
    plt.semilogy(np.arange(d,NC+d,d)[::q],np.mean(timing[::q, 1,:],1),'--',label= 'Updating time')
    # plt.semilogy(np.arange(d,NC+d,d)[::q],timing[::q, 2,0],'--',label= 'Recomputing SVD time')
    plt.legend()
    plt.xlabel('Columns added')
    plt.ylabel('time (seconds)')
    plt.show()
    prediction = prediction[:-1, :, :]


    plt.figure()

    plt.plot(np.arange(d,NC,d), np.mean(prediction[:, 0,:],1), label='folding-in predictions')
    plt.plot(np.arange(d, NC , d), np.mean(prediction[:, 1, :], 1), label='Updating predictions')
    # plt.plot(np.arange(d, NC , d), np.mean(prediction[:, 2, :], 1), label='Recomputing predictions')
    plt.plot(np.arange(d, NC , d), np.mean(prediction[:, 3, :], 1), label='Mean')
    plt.plot(np.arange(d, NC , d), np.mean(prediction[:, 4, :], 1),'--', label='Observation')

    plt.legend()
    plt.xlabel('Columns added')
    plt.ylabel('time (seconds)')
    plt.show()

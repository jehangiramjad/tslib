######################################################
#
# The Time Series Model based on SVD
#
######################################################
import sys, os
sys.path.append("..")
sys.path.append(os.getcwd())

import copy
import numpy as np
import pandas as pd
from algorithms.svdWrapper import SVDWrapper as SVD

import tsUtils

class SVDModel:


    # seriesToPredictKey:       (string) the time series of interest (key)
    # kSingularValuesToKeep:    (int) the number of singular values to retain
    # N:                        (int) the number of rows of the matrix for each series
    # M:                        (int) the number of columns for the amtrix for each series
    # probObservation:          (float) the independent probability of observation of each entry in the matrix
    # svdMethod:                (string) the SVD method to use (optional)
    # otherSeriesKeysArray:     (array) an array of keys for other series which will be used to predict 
    # includePastDataOnly:      (Boolean) defaults to True. If this is set to False, 
    #                               the time series in 'otherSeriesKeysArray' will include the latest data point.
    #                               Note: the time series of interest (seriesToPredictKey) will never include 
    #                               the latest data-points for prediction
    def __init__(self, seriesToPredictKey, kSingularValuesToKeep, N, M, probObservation=1.0, svdMethod='numpy', otherSeriesKeysArray=[], includePastDataOnly=True):

        self.seriesToPredictKey = seriesToPredictKey
        self.otherSeriesKeysArray = otherSeriesKeysArray
        self.includePastDataOnly = includePastDataOnly

        self.N = N
        self.M = M

        self.kSingularValues = kSingularValuesToKeep
        self.svdMethod = svdMethod

        self.Uk = None
        self.Vk = None
        self.sk = None
        self.matrix = None
        self.lastRowObservations = None

        self.p = probObservation

        self.weights = None
        self.div = 1.0
        self.diff = 0.0

    # run a least-squares regression of the last row of self.matrix and all other rows of self.matrix
    # sets and returns the weights
    # DO NOT call directly
    def _computeWeights(self):
        
        if (self.lastRowObservations is None):
            raise Exception('Do not call _computeWeights() directly. It should only be accessed via class methods.')

        # need to decide how to produce weights based on whether the N'th data points are to be included for the other time series or not
        # for the seriesToPredictKey we only look at the past. For others, we could be looking at the current data point in time as well.
        
        matrixDim1 = (self.N * len(self.otherSeriesKeysArray)) + self.N-1
        matrixDim2 = np.shape(self.Uk)[1]# this is the number of singular values selected
        eachTSRows = self.N

        
        if (self.includePastDataOnly == True):
            matrixDim1 = ((self.N - 1) * len(self.otherSeriesKeysArray)) + self.N-1
            eachTSRows = self.N - 1
            U = np.zeros([matrixDim1, matrixDim2])

            i = 0
            j = 0
            while (i < matrixDim1):
                U[i : i+ eachTSRows, :] = self.Uk[j : j + self.N - 1, : ] 

                i += eachTSRows
                j += self.N
        else:
            U = self.Uk[0:-1, :]

        matrixInverse = tsUtils.pInverseMatrixFromSVD(self.sk, U, self.Vk, probability=self.p)
        self.weights = np.dot(matrixInverse.T, (1.0/self.p) * self.lastRowObservations.T)

    def denoisedDF(self):
        setAllKeys = set(self.otherSeriesKeysArray)
        setAllKeys.add(self.seriesToPredictKey)

        single_ts_rows = self.N
        matrix_cols = self.M
        matrix_rows = len(setAllKeys) * single_ts_rows

        dataDict = {}
        rowIndex = 0
        for key in self.otherSeriesKeysArray:

            dataDict.update({key: self.matrix[rowIndex*single_ts_rows: (rowIndex+1)*single_ts_rows, :].flatten('F')})
            rowIndex += 1

        dataDict.update({self.seriesToPredictKey: self.matrix[rowIndex*single_ts_rows: (rowIndex+1)*single_ts_rows, :].flatten('F')})

        return pd.DataFrame(data=dataDict)


    # keyToSeriesDictionary: (Pandas dataframe) a key-value Series (time series)
    # Note that the keys provided in the constructor MUST all be present
    # The values must be all numpy arrays of floats.
    # This function sets the "de-noised" and imputed data matrix which can be accessed by the .matrix attribute
    # NOTE: ensure that all timeseries in keyToSeriesDF are normalized to lie between [-1, 1].
    def fit(self, keyToSeriesDF):

        setAllKeys = set(self.otherSeriesKeysArray)
        setAllKeys.add(self.seriesToPredictKey)

        if (len(set(keyToSeriesDF.columns.values).intersection(setAllKeys)) != len(setAllKeys)):
            raise Exception('keyToSeriesDF does not contain ALL keys provided in the constructor.')

        #if ((np.nanmax(keyToSeriesDF) > 1.0) | (np.nanmin(keyToSeriesDF) < -1.0)):
        #    raise Exception('All time series must lie within [-1, 1]')

        # impute with the least informative value
        max = np.nanmax(keyToSeriesDF)
        min = np.nanmin(keyToSeriesDF)
        diff = 0.5*(min + max)
        keyToSeriesDF = keyToSeriesDF.fillna(value=diff)

        T = self.N * self.M
        for key in setAllKeys:
            if (len(keyToSeriesDF[key]) < T):
                raise Exception('All series (columns) provided must have length >= %d' %T)


        # initialize the matrix of interest
        single_ts_rows = self.N
        matrix_cols = self.M
        matrix_rows = (len(setAllKeys) * single_ts_rows)

        self.matrix = np.zeros([matrix_rows, matrix_cols])

        seriesIndex = 0
        for key in self.otherSeriesKeysArray: # it is important to use the order of keys set in the model
            self.matrix[seriesIndex*single_ts_rows: (seriesIndex+1)*single_ts_rows, :] = tsUtils.arrayToMatrix(keyToSeriesDF[key][-1*T:].values, single_ts_rows, matrix_cols)
            seriesIndex += 1

        # finally add the series of interest at the bottom
       # tempMatrix = tsUtils.arrayToMatrix(keyToSeriesDF[self.seriesToPredictKey][-1*T:].values, self.N, matrix_cols)
        self.matrix[seriesIndex*single_ts_rows: (seriesIndex+1)*single_ts_rows, :] = tsUtils.arrayToMatrix(keyToSeriesDF[self.seriesToPredictKey][-1*T:].values, single_ts_rows, matrix_cols)
        
        # set the last row of observations
        self.lastRowObservations = copy.deepcopy(self.matrix[-1, :])

        # now produce a thresholded/de-noised matrix. this will over-write the original data matrix
        svdMod = SVD(self.matrix, method='numpy')
        (self.sk, self.Uk, self.Vk) = svdMod.reconstructMatrix(self.kSingularValues, returnMatrix=False)

        self.matrix = tsUtils.matrixFromSVD(self.sk, self.Uk, self.Vk, probability=self.p)

        # set weights
        self._computeWeights()



    # otherKeysToSeriesDFNew:     (Pandas dataframe) needs to contain all keys provided in the model;
    #                           If includePastDataOnly was set to True (default) in the model, then:
    #                               each series/array MUST be of length >= self.N - 1
    #                               If longer than self.N - 1, then the most recent self.N - 1 points will be used
    #                           If includePastDataOnly was set to False in the model, then:
    #                               all series/array except seriesToPredictKey MUST be of length >= self.N (i.e. includes the current), 
    #                               If longer than self.N, then the most recent self.N points will be used
    #
    # predictKeyToSeriesDFNew:   (Pandas dataframe) needs to contain the seriesToPredictKey and self.N - 1 points past points.
    #                           If more points are provided, the most recent self.N - 1 points are selected.   
    #
    # bypassChecks:         (Boolean) if this is set to True, then it is the callee's responsibility to provide
    #                           all required series of appropriate lengths (see above).
    #                           It is advised to leave this set to False (default).         
    def predict(self, otherKeysToSeriesDFNew, predictKeyToSeriesDFNew, bypassChecks=False):

        nbrPointsNeeded = self.N - 1
        if (self.includePastDataOnly == False):
            nbrPointsNeeded = self.N

        if (bypassChecks == False):

            if (self.weights is None):
                raise Exception('Before predict() you need to call "fit()" on the model.')

            if (len(set(otherKeysToSeriesDFNew.columns.values).intersection(set(self.otherSeriesKeysArray))) < len(set(self.otherSeriesKeysArray))):
                raise Exception('keyToSeriesDFNew does not contain ALL keys provided in the constructor.')

            for key in self.otherSeriesKeysArray:
                points = len(otherKeysToSeriesDFNew[key])
                if (points < nbrPointsNeeded):
                    raise Exception('Series (%s) must have length >= %d' %(key, nbrPointsNeeded))

            points = len(predictKeyToSeriesDFNew[self.seriesToPredictKey])
            if (points < self.N - 1):
                raise Exception('Series (%s) must have length >= %d' %(self.seriesToPredictKey, self.N - 1))

        newDataArray = np.zeros((len(self.otherSeriesKeysArray) * nbrPointsNeeded) + self.N - 1)
        indexArray = 0
        for key in self.otherSeriesKeysArray:
            newDataArray[indexArray: indexArray + nbrPointsNeeded] = otherKeysToSeriesDFNew[key][0:nbrPointsNeeded].values

            indexArray += nbrPointsNeeded

        # at last fill in the time series of interest
        newDataArray[indexArray:] = predictKeyToSeriesDFNew[self.seriesToPredictKey][0: self.N - 1].values

        # dot product
        return np.dot(self.weights, newDataArray)


####################################################
# Testing

# seriesToPredictKey = 'a1'
# kSingularValuesToKeep = 4
# N = 4
# M = 5
# p = 1.0
# svdMethod='numpy'
# otherSeriesKeysArray=['a2', 'a3']
# includePastDataOnly = False

# keytoSeriesDictionary = {'a1': np.random.normal(0, 1, N*M), 'a2': np.random.normal(0, 1, N*M), 'a3':np.random.normal(0, 1, N*M)}
# df = pd.DataFrame(data=keytoSeriesDictionary)
# mod = SVDModel(seriesToPredictKey, kSingularValuesToKeep, N, M, probObservation=1.0, svdMethod='numpy', otherSeriesKeysArray=otherSeriesKeysArray, includePastDataOnly=includePastDataOnly)
# mod.fit(df)

# # predict
# new = {'a1': np.random.normal(0, 1, N), 'a2': np.random.normal(0, 1, N), 'a3':np.random.normal(0, 1, N)}
# dfNew = pd.DataFrame(data=new)
# res = mod.predict(dfNew, bypassChecks=False)



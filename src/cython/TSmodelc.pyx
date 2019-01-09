import numpy as np
import pandas as pd
from  tslib.src.models.tsSVDModelc import SVDModel

cdef class  TSmodelc(object):
    cdef int T, L,T0, TimeSeriesIndex, rectFactor, MUpdateIndex, kSingularValuesToKeep, ReconIndex
    cdef float gamma
    def __cinit__(self, int kSingularValuesToKeep,  int T=int(1e4), float gamma=0.2, int T0=1000, int NoRows=False, int rectFactor=10):
        self.kSingularValuesToKeep = kSingularValuesToKeep
        if NoRows:
            self.T = int(T ** 2 * rectFactor)
            self.L = int(T)
        else:
            self.T = T
            self.L = np.sqrt(T)
        self.gamma = gamma
        self.T0 = T0
        self.TimeSeriesIndex = 0
        self.ReconIndex = 0
        self.rectFactor = rectFactor
        self.MUpdateIndex = 0

class TSmodel(TSmodelc):
    # kSingularValuesToKeep:    (int) the number of singular values to retain
    # T0:                       (int) the number of entries below which the model will simply give the average.
    # T:                        (int) the maximum number of entries for each SVD model
    # gamma:                    (float) (0,1) fraction of T after which the model is updated

    def __init__(self, kSingularValuesToKeep, T=int(1e4), gamma=0.2, T0=1000, NoRows=False, rectFactor=10):
        TSmodelc.__cinit__(kSingularValuesToKeep, T, gamma, T0, NoRows, rectFactor)
        self.models = {}
        self.TimeSeries = None


    def UpdateModel(self, NewEntries):
        cdef int ML = len(NewEntries)/self.T
        cdef int i = 0
        for i in range(ML):
            self.updateTS(NewEntries[i * self.T: (i+1) * self.T])
            self.fitModels()
        cdef int UpdateChunk =  self.L
        NewEntries = NewEntries[(i + 1) * self.T:]
        for i in range(int(len(NewEntries)/(UpdateChunk))):
            self.updateTS( NewEntries[i * UpdateChunk: (i+1) * UpdateChunk])
            self.fitModels()

    def updateTS(self, NewEntries):
        # Update the time series with the new entries.
        # only keep the last T entries
        cdef int n = len(NewEntries)
        self.TimeSeriesIndex = self.TimeSeriesIndex + n
        if self.TimeSeriesIndex == n:
            self.TimeSeries = NewEntries

        elif len(self.TimeSeries) < self.T:
            TSarray = np.zeros(len(self.TimeSeries) + n)
            TSarray[:len(self.TimeSeries)] = self.TimeSeries
            TSarray[len(self.TimeSeries):] = NewEntries
            self.TimeSeries = TSarray

        else:
            if n < self.T: self.TimeSeries[:self.T - n] = self.TimeSeries[-self.T + n:]
            self.TimeSeries[-n:] = NewEntries

        if len(self.TimeSeries) > self.T:
            self.TimeSeries = self.TimeSeries[-self.T:]




    def fitModels(self):
        cdef int N, M, ModelLength
        # Determine which model to fit
        cdef int ModelIndex = max((self.TimeSeriesIndex - 1) / (self.T / 2) - 1, 0)
        # fit/update New/existing Model or do nothing
        cdef int start = ModelIndex * self.T / 2
        cdef int  lenEntriesSinceCons = self.TimeSeriesIndex - self.ReconIndex

        if self.TimeSeriesIndex < self.T0:
            pass
        if lenEntriesSinceCons > self.T:
            print lenEntriesSinceCons, self.T
            raise Exception('Model should be updated before T values are assigned')
        else:
            if ModelIndex not in self.models:

                initEntries = self.TimeSeries[
                              (self.T / 2) - self.TimeSeriesIndex % (self.T / 2): self.T - self.TimeSeriesIndex % (
                              self.T / 2)]
                if ModelIndex != 0: assert len(initEntries) == self.T / 2
                rect = 2
                if lenEntriesSinceCons == self.T:

                    initEntries = self.TimeSeries[:self.T]
                    rect = 1

                N = int(np.sqrt(len(initEntries) / (self.rectFactor / rect)))
                M = len(initEntries) / N
                self.ReconIndex = N * M + start
                self.models[ModelIndex] = SVDModel('t1', self.kSingularValuesToKeep, N, M, start=start)
                self.models[ModelIndex].fit(pd.DataFrame(data={'t1': initEntries}))
                self.MUpdateIndex = self.ReconIndex

                if lenEntriesSinceCons == self.T:
                    return
            Model = self.models[ModelIndex]

            lenEntriesSinceCons = self.TimeSeriesIndex - self.ReconIndex
            ModelLength = Model.N * Model.M + start
            if (float(lenEntriesSinceCons) / (self.ReconIndex - Model.start) >= self.gamma) or (
                            self.TimeSeriesIndex % (self.T / 2) == 0):  # condition to create new model

                TSlength = self.TimeSeriesIndex - start
                N = int(np.sqrt(TSlength / self.rectFactor))
                M = TSlength / N
                TSeries = self.TimeSeries[-TSlength:]
                TSeries = TSeries[:N * M]
                self.models[ModelIndex] = SVDModel('t1', self.kSingularValuesToKeep, N, M, start=start,
                                                   TimesReconstructed=Model.TimesReconstructed + 1,
                                                   TimesUpdated=Model.TimesUpdated)

                self.models[ModelIndex].fit(pd.DataFrame(data={'t1': TSeries}))
                self.ReconIndex = N * M + start
                self.MUpdateIndex = self.ReconIndex

            else:

                Model = self.models[ModelIndex]
                N = Model.N
                if self.TimeSeriesIndex - ModelLength < N:
                    pass
                else:
                    D = self.TimeSeries[-(self.TimeSeriesIndex - ModelLength):]
                    p = len(D) / N
                    D = D[:N * p]
                    Model.updateSVD(D, 'UP')
                    self.MUpdateIndex = Model.N * Model.M + Model.start

    def denoiseTS(self, index=None, range=True):

        if range or index is None:
            if index is None:
                index = [0, self.ReconIndex]
            denoised = np.zeros(index[1] - index[0])
            count = np.zeros(index[1] - index[0])
            y1, y2 = index[0], index[1]
            for Model in self.models.values():
                x1, x2 = Model.start, Model.M * Model.N + Model.start
                if x1 <= y2 and y1 <= x2:
                    RIndex = np.array([max(x1, y1), min(x2, y2)])
                    RIndexS = RIndex - y1

                    denoised[RIndexS[0]:RIndexS[1]] += Model.denoisedTS(RIndex - x1, range)
                    count[RIndexS[0]:RIndexS[1]] += 1
            denoised[count == 0] = np.nan
            denoised[count > 0] = denoised[count > 0] / count[count > 0]
            return denoised

        else:

            index = np.array(index)
            I = len(index)
            denoised = np.zeros(I)
            models = np.zeros(2 * I)
            models[:I] = (index) / (self.T / 2) - 1
            models[I:] = models[:I] + 1
            models[models < 0] = 0
            count = np.zeros(len(index))

            for ModelNumber in np.unique(models):
                Model = self.models[ModelNumber]
                x1, x2 = Model.start, Model.M * Model.N + Model.start
                updatedIndices = np.logical_and(index >= x1, index < x2)
                assert np.sum(updatedIndices) > 0
                count += updatedIndices
                denoised[updatedIndices] += Model.denoisedTS(index[updatedIndices] - x1, range)

            denoised[count == 0] = np.nan
            denoised[count > 0] = denoised[count > 0] / count[count > 0]
            return denoised

    def predict(self, index=None, method='average', NoModels=None):
        if NoModels is None: NoModels = len(self.models)
        # if index next predict
        if index is None or index == self.TimeSeriesIndex + 1:

            TSDF = pd.DataFrame(data={'t1': self.TimeSeries[-self.L:]})
            UsedModels = [a[1] for a in sorted(self.models.items(), key=lambda pair: pair[0])][-NoModels:]
            predicions = np.array([mod.predict(pd.DataFrame(data={}), TSDF) for mod in UsedModels])
            return np.mean(predicions)

        else:
            return 0
            # if not predict till then
            # get models weight and average all predictions

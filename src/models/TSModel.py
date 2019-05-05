import numpy as np
import pandas as pd
from  tslib.src.models.tsSVDModel import SVDModel
from math import ceil
from tslib.src.database.DBInterface import Interface


class TSmodel(object):
    # kSingularValuesToKeep:    (int) the number of singular values to retain
    # T0:                       (int) the number of entries below which the model will not be trained.
    # T:                        (int) Number of entries in each submodel
    # gamma:                    (float) (0,1) fraction of T after which the model is updated
    # rectFactor:               (int) the ration of no. columns to the number of rows in each sub-model

    def __init__(self, kSingularValuesToKeep, T=int(1e5), gamma=0.2, T0=1000, rectFactor=10, interface=Interface
                 , time_series_table = ['ts_test','ts','row_id'], model_table_name='test'):
        self.kSingularValuesToKeep = kSingularValuesToKeep
        self.T = int(T)
        self.L = int(np.sqrt(T / rectFactor))
        self.gamma = gamma
        self.models = {}
        self.T0 = T0
        self.TimeSeries = None
        self.TimeSeriesIndex = 0
        self.ReconIndex = 0
        self.rectFactor = rectFactor
        self.MUpdateIndex = 0
        self.db_interface = interface
        self.model_tables_name = model_table_name
        self.time_series_table = time_series_table

    def get_model_index(self, ts_index=None):
        if ts_index is None:
            ts_index = self.TimeSeriesIndex
        model_index = max((ts_index - 1) / (self.T / 2) - 1, 0)
        return model_index

    def create_index(self):
        pass

    def update_index(self):
        new_entries = self.get_range(self.TimeSeriesIndex)

        if len(new_entries)>0:
            self.update_model(new_entries)
        # check if there is un_written models
        self.WriteModel()



    def update_model(self, NewEntries):

        if len(self.models) == 1 and len(NewEntries) < self.T / 2:
            UpdateChunk = 20 * int(np.sqrt(self.T0))
        else:
            UpdateChunk = int(self.T / (2 * self.rectFactor) * 0.85)

        # find if new models should be constructed
        N = len(NewEntries)
        current_no_models = len(self.models)
        updated_no_models = self.get_model_index(self.TimeSeriesIndex + N) + 1

        # if no new models are to be constructed
        if current_no_models == updated_no_models:
            # No new models to be constructed
            # If it is a big update, do it at once
            last_model_size = self.models[updated_no_models - 1].M * self.models[updated_no_models - 1].N
            if len(NewEntries) / float(last_model_size) > self.gamma:
                self.updateTS(NewEntries[:])
                self.fitModels()
                return
                # Otherwise, update it chunk by chunk (Because Incremental Update requires small updates (not really need to be fixed))
            i = -1
            for i in range(int(ceil(len(NewEntries) / (UpdateChunk)))):
                self.updateTS(NewEntries[i * UpdateChunk: (i + 1) * UpdateChunk])
                self.fitModels()

        else:
            if current_no_models > 0:
                fillFactor = (self.TimeSeriesIndex % (self.T / 2))
                FillElements = (self.T / 2 - fillFactor) * (fillFactor > 0)
                if FillElements > 0:
                    self.updateTS(NewEntries[:FillElements])
                    print self.TimeSeriesIndex
                    self.fitModels()
                    NewEntries = NewEntries[FillElements:]

            SkipNext = False
            for i in range(updated_no_models - current_no_models + (current_no_models == 0)):
                if SkipNext:
                    SkipNext = False
                    continue
                if len(self.models) == 0:
                    self.updateTS(NewEntries[: self.T])
                    SkipNext = True
                    self.fitModels()
                    i += 1
                else:
                    self.updateTS(NewEntries[i * (self.T / 2): (i + 1) * (self.T / 2)])
                    self.fitModels()

    def updateTS(self, NewEntries):
        # Update the time series with the new entries.
        # only keep the last T entries

        n = len(NewEntries)

        if n > self.T / 2 and len(self.models) > 1:
            print n, self.T
            raise Exception('TimeSeries should be updated before T/2 values are assigned')

        self.TimeSeriesIndex += n

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

        # Determine which model to fit
        ModelIndex = self.get_model_index(self.TimeSeriesIndex)
        # fit/update New/existing Model or do nothing
        lenEntriesSinceCons = self.TimeSeriesIndex - self.ReconIndex
        lenEntriesSinceLastUpdate = self.TimeSeriesIndex - self.MUpdateIndex
        if self.TimeSeriesIndex < self.T0:
            return
        if lenEntriesSinceLastUpdate > self.T and ModelIndex != 0:
            print self.TimeSeriesIndex, self.MUpdateIndex, [(m.N, m.M, m.start) for m in self.models.values()]
            raise Exception('Model should be updated before T values are assigned')

        if ModelIndex not in self.models:

            initEntries = self.TimeSeries[(self.T / 2) - self.TimeSeriesIndex % (self.T / 2): self.T - self.TimeSeriesIndex %(self.T / 2)]

            initEntries = self.TimeSeries[(self.T / 2) - self.TimeSeriesIndex % (self.T / 2):]

            start = self.TimeSeriesIndex - self.TimeSeriesIndex % (self.T / 2) - self.T / 2

            # if ModelIndex != 0: assert len(initEntries) == self.T / 2
            rect = 1

            if lenEntriesSinceCons == self.T / 2 or ModelIndex == 0:
                initEntries = self.TimeSeries[:]
                start = max(self.TimeSeriesIndex - self.T, 0)

            N = int(np.sqrt(len(initEntries) / (self.rectFactor / rect)))
            M = len(initEntries) / N
            self.ReconIndex = N * M + start
            self.models[ModelIndex] = SVDModel('t1', self.kSingularValuesToKeep, N, M, start=start)
            self.models[ModelIndex].fit(pd.DataFrame(data={'t1': initEntries}))
            self.MUpdateIndex = self.ReconIndex

            if lenEntriesSinceCons == self.T / 2 or ModelIndex == 0:

                return
        Model = self.models[ModelIndex]

        lenEntriesSinceCons = self.TimeSeriesIndex - self.ReconIndex
        ModelLength = Model.N * Model.M + Model.start
        if (float(lenEntriesSinceCons) / (self.ReconIndex - Model.start) >= self.gamma) or (
                        self.TimeSeriesIndex % (self.T / 2) == 0):  # condition to create new model

            TSlength = self.TimeSeriesIndex - Model.start
            N = int(np.sqrt(TSlength / self.rectFactor))
            M = TSlength / N
            TSeries = self.TimeSeries[-TSlength:]
            TSeries = TSeries[:N * M]

            self.models[ModelIndex] = SVDModel('t1', self.kSingularValuesToKeep, N, M, start=Model.start,
                                               TimesReconstructed=Model.TimesReconstructed + 1,
                                               TimesUpdated=Model.TimesUpdated)

            self.models[ModelIndex].fit(pd.DataFrame(data={'t1': TSeries}))
            self.ReconIndex = N * M + Model.start
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
                Model.updateSVD(D)
                self.MUpdateIndex = Model.N * Model.M + Model.start



    def WriteModel(self):
        """
        - Should not take db info
        """
        ModelName = self.model_tables_name
        if len(self.models) == 0:
            return
        N = self.L
        M = N * self.rectFactor
        tableNames = [ModelName + '_' + c for c in ['u', 'v', 's', 'c']]
        U_table = np.zeros(
            [(len(self.models) - 1) * N + self.models[len(self.models) - 1].N, 1 + self.kSingularValuesToKeep])
        for i, m in self.models.items():
            if i == len(self.models) - 1:
                U_table[i * N:, 1:] = m.Uk
                U_table[i * N:, 0] = int(i)
            else:
                U_table[i * N:(i + 1) * N, 1:] = m.Uk
                U_table[i * N:(i + 1) * N, 0] = int(i)

        columns = ['modelno'] + ['u' + str(i) for i in range(1, self.kSingularValuesToKeep + 1)]
        udf = pd.DataFrame(columns=columns, data=U_table)
        # udf['tsrow'] = (udf.index - 0.5 * N * udf['modelno']).astype(int)
        udf['tsrow'] = (udf.index % N).astype(int)

        self.db_interface.create_table( tableNames[0],udf, 'row_id', index_label='row_id')
        # self.writeTable(udf, tableNames[0], host, database, user, password)

        V_table = np.zeros(
            [(len(self.models) - 1) * M + self.models[len(self.models) - 1].M, 1 + self.kSingularValuesToKeep])
        for i, m in self.models.items():
            if i == len(self.models) - 1:
                V_table[i * M:, 1:] = m.Vk
                V_table[i * M:, 0] = int(i)
            else:
                V_table[i * M:(i + 1) * M, 1:] = m.Vk
                V_table[i * M:(i + 1) * M, 0] = int(i)
        columns = ['modelno'] + ['v' + str(i) for i in range(1, self.kSingularValuesToKeep + 1)]
        vdf = pd.DataFrame(columns=columns, data=V_table)
        vdf['tscolumn'] = (vdf.index - 0.5 * M * vdf['modelno']).astype(int)
        self.db_interface.create_table(tableNames[1],vdf,  'row_id', index_label='row_id')


        s_table = np.zeros([len(self.models), 1 + self.kSingularValuesToKeep])
        for i, m in self.models.items():
            s_table[i, 1:] = m.sk
            s_table[i, 0] = int(i)
        columns = ['modelno'] + ['s' + str(i) for i in range(1, self.kSingularValuesToKeep + 1)]
        sdf = pd.DataFrame(columns=columns, data=s_table)

        self.db_interface.create_table( tableNames[2], sdf, 'row_id', index_label='row_id')


        id_c = 0
        W = len(self.models[0].weights)
        c_table = np.zeros([len(self.models) * W, 3])
        for i, m in self.models.items():
            coeNu = 0
            for w in m.weights:
                c_table[id_c, :] = [i, coeNu, w]
                id_c += 1
                coeNu += 1
        cdf = pd.DataFrame(columns=['modelno', 'coeffpos', 'coeffvalue'], data=c_table)
        self.db_interface.create_table(tableNames[3], cdf, 'row_id', index_label='row_id')

        lasModel = len(self.models) - 1
        self.db_interface.create_index(tableNames[0], 'tsrow')
        self.db_interface.create_index(tableNames[0], 'modelno')
        self.db_interface.create_index(tableNames[1], 'tscolumn')
        self.db_interface.create_index(tableNames[2], 'modelno')
        self.db_interface.create_index(tableNames[3], 'modelno')
        self.db_interface.create_index(tableNames[3], 'coeffpos')


        self.db_interface.create_coefficients_average_table(tableNames[3] , tableNames[3] + '_view', [10,20,100],
                                                            lasModel)


    def denoiseTS(self, index=None, range=True):
        if range or index is None:
            if index is None:
                index = [0, self.MUpdateIndex]
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

    def predict(self, index=None, method='average', NoModels=None, dataPoints=None):

        if NoModels is None: NoModels = len(self.models)
        # if index next predict

        if dataPoints is None and (index is None or index == self.TimeSeriesIndex + 1):
            TSDF = pd.DataFrame(data={'t1': self.TimeSeries[-self.L:]})

            UsedModels = [a for a in self.models.values()[-NoModels:]]
            predicions = np.array([mod.predict(pd.DataFrame(data={}), TSDF) for mod in UsedModels])
            return np.mean(predicions)

        elif dataPoints is None and index <= self.TimeSeriesIndex:
            slack = self.TimeSeriesIndex - index + 1
            if slack > (self.T - self.L): raise Exception
            TSDF = pd.DataFrame(data={'t1': self.TimeSeries[-self.L - slack:-slack]})
            UsedModels = [a[1] for a in self.models.items()[-NoModels - 1:-1]]
            predicions = np.array([mod.predict(pd.DataFrame(data={}), TSDF) for mod in UsedModels])
            return np.mean(predicions)

        elif dataPoints is not None:
            TSDF = pd.DataFrame(data={'t1': dataPoints})
            UsedModels = [a for a in self.models.values()[-NoModels:]]
            predicions = np.array([mod.predict(pd.DataFrame(data={}), TSDF) for mod in UsedModels])
            return np.mean(predicions)
        else:
            return 0
            # if not predict till then
            # get models weight and average all predictions
    def get_prediction(self, t):
        """
        call get_imp or get_fore depending on t
        """
        if t > (self.TimeSeriesIndex-1):
            return self.get_forecast(t)
        else:
            return self.get_imputation(t)



    def get_imputation(self, t):
        """
        implement the same singles point query. use get from table function in interface
        """

        modelNo = self.get_model_index(t+1)
        N = self.models[modelNo].N
        tscolumn = t/ N
        tsrow = t%N
        U = self.db_interface.get_U_row(self.model_tables_name+'_u', [tsrow,tsrow], [modelNo,modelNo+1],self.kSingularValuesToKeep)
        V = self.db_interface.get_V_row(self.model_tables_name + '_v', [tscolumn, tscolumn], self.kSingularValuesToKeep)
        S = self.db_interface.get_S_row(self.model_tables_name + '_s', [modelNo,modelNo+1], self.kSingularValuesToKeep)
        U1 = U[0, :]
        V1 = V[0, :]
        S1 = S[0, :]
        if len(S) == 2 and len(V) ==2 and len(U) ==2:
            U2 = U[1, :]
            V2 = V[1, :]
            S2 = S[1, :]
            return 0.5*(np.dot(U1*S1,V1.T) + np.dot(U2*S2,V2.T))
        return np.dot(U1*S1,V1.T)

    def get_forecast(self, t):
        """
        implement the same singles point query. use get from table function in interface
        """
        coeffs = self.db_interface.get_coeff( self.model_tables_name+'_c_view', 'average')
        no_coeff = len(coeffs)
        last_obs = self.db_interface.get_time_series(self.time_series_table[0],t-no_coeff, t-1, value_column=self.time_series_table[1], index_col=self.time_series_table[2], Desc = False)

        return np.dot(coeffs[:len(last_obs)].T, last_obs)


    def get(self, t):
        """
        implement the same singles point query. use get from table function in interface
        """

        return self.db_interface.get_time_series(self.time_series_table[0], t , t,
                                                     value_column=self.time_series_table[1],
                                                     index_col=self.time_series_table[2])


    def get_range(self, t1 , t2 = None):
        """
        implement the same singles point query. use get from table function in interface
        """

        return self.db_interface.get_time_series(self.time_series_table[0], t1, t2,
                                                 value_column=self.time_series_table[1],
                                                 index_col=self.time_series_table[2])[:,0]


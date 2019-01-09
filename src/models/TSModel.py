import numpy as np
import pandas as pd
import psycopg2
from  tslib.src.models.tsSVDModel import SVDModel
from sqlalchemy import create_engine
import io
class TSmodel(object):
    # kSingularValuesToKeep:    (int) the number of singular values to retain
    # T0:                       (int) the number of entries below which the model will simply give the average.
    # T:                        (int) the maximum number of entries for each SVD model
    # gamma:                    (float) (0,1) fraction of T after which the model is updated

    def __init__(self, kSingularValuesToKeep, T=int(1e4), gamma=0.2, T0=1000, NoRows=False, rectFactor=10):
        self.kSingularValuesToKeep = kSingularValuesToKeep
        if NoRows:
            self.T = int(T ** 2 * rectFactor)
            self.L = int(T)
        else:
            self.T = T
            self.L = np.sqrt(T)
        self.gamma = gamma
        self.models = {}
        self.T0 = T0
        self.TimeSeries = None
        self.TimeSeriesIndex = 0
        self.ReconIndex = 0
        self.rectFactor = rectFactor
        self.MUpdateIndex = 0

    def UpdateModel(self, NewEntries, patch = False):
        ML = len(NewEntries)/(self.T/2)
        i = -1
        if ML <=1 : ML = 2
        for i in range(1,ML):
            if i == 1:
                self.updateTS(NewEntries[: self.T])
            else:
                self.updateTS(NewEntries[i * (self.T/2): (i+1) * (self.T/2)])
            self.fitModels()

        if patch: return

        if len(self.models) == 0: UpdateChunk = int(np.sqrt(self.T0))
        else:    UpdateChunk = self.L

        NewEntries = NewEntries[(i + 1) * (self.T/2):]
        for i in range(int(len(NewEntries)/(UpdateChunk))):

            self.updateTS( NewEntries[i * UpdateChunk: (i+1) * UpdateChunk])
            self.fitModels()

    def updateTS(self, NewEntries):
        # Update the time series with the new entries.
        # only keep the last T entries

        n = len(NewEntries)

        if n > self.T / 2 and len(self.models) > 1:
            print n, self.T
            raise Exception('TimeSeries should be updated before T/2 values are assigned')

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

        # Determine which model to fit
        ModelIndex = max((self.TimeSeriesIndex - 1) / (self.T / 2) - 1, 0)
        # fit/update New/existing Model or do nothing

        lenEntriesSinceCons = self.TimeSeriesIndex - self.ReconIndex

        if self.TimeSeriesIndex < self.T0:
            pass
        if lenEntriesSinceCons > self.T/2 and ModelIndex != 0:

            print lenEntriesSinceCons, self.T
            raise Exception('Model should be updated before T/2 values are assigned')

        if ModelIndex not in self.models:

            initEntries = self.TimeSeries[
                          (self.T / 2) - self.TimeSeriesIndex % (self.T/2): self.T - self.TimeSeriesIndex %
                                                                                     (self.T/2)]
            start = self.TimeSeriesIndex - self.TimeSeriesIndex % (self.T / 2)
            if ModelIndex != 0: assert len(initEntries) == self.T / 2
            rect = 1
            if lenEntriesSinceCons == self.T/2 or ModelIndex == 0:
                initEntries = self.TimeSeries[:]
                start = max(self.TimeSeriesIndex - self.T, 0)
            N = int(np.sqrt(len(initEntries) / (self.rectFactor / rect)))
            M = len(initEntries) / N
            self.ReconIndex = N * M + start
            self.models[ModelIndex] = SVDModel('t1', self.kSingularValuesToKeep, N, M, start=start)
            self.models[ModelIndex].fit(pd.DataFrame(data={'t1': initEntries}))
            self.MUpdateIndex = self.ReconIndex

            if lenEntriesSinceCons == self.T/2 or ModelIndex == 0:
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
            print len(TSeries), ModelIndex,  Model.start, self.models[ModelIndex-1].start
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

    def writeTable(self, df, tableName, host="localhost", database="querytime_test", user="aalomar", password="AAmit32lids"):
        engine = create_engine('postgresql+psycopg2://' + user + ':'+ password+ '@'+ host+ '/' + database)

        df.head(0).to_sql(tableName , engine, if_exists='replace', index=True, index_label = 'rowID')  # truncates the table
        conn = engine.raw_connection()
        cur = conn.cursor()
        output = io.BytesIO()
        df.to_csv(output, sep='\t', header = False, index = True, index_label = 'rowID')
        output.seek(0)
        cur.copy_from(output, tableName, null="")  # null values become ''
        conn.commit()

    def WriteModel(self, ModelName = 'model', host = "localhost",database="querytime_test", user="aalomar", password="AAmit32lids"):
        if len(self.models) == 0:
            return
        N = self.L
        M = N*self.rectFactor
        tableNames = [ModelName+'_' + c for c in ['u', 'v', 's', 'c']]
        U_table = np.zeros([(len(self.models)-1) * N + self.models[len(self.models)-1].N , 1 + self.kSingularValuesToKeep])
        for i, m in self.models.items():
            if i == len(self.models)-1:
                U_table[i * N:, 1:] = m.Uk
                U_table[i * N:, 0] = int(i)
            else:
                U_table[i * N:(i + 1) * N, 1:] = m.Uk
                U_table[i * N:(i + 1) * N, 0] = int(i)
        columns =['modelno'] +['u' + str(i) for i in range(1, self.kSingularValuesToKeep + 1)]
        udf = pd.DataFrame(columns=columns, data=U_table)
        udf['tsrow'] = (udf.index - 0.5 * N * udf['modelno']).astype(int)
        self.writeTable(udf, tableNames[0], host, database, user, password)

        V_table = np.zeros([(len(self.models)-1) * M + self.models[len(self.models)-1].M, 1 + self.kSingularValuesToKeep])
        for i, m in self.models.items():
            if i == len(self.models)-1:
                V_table[i * M:, 1:] = m.Vk
                V_table[i * M:, 0] = int(i)
            else:
                V_table[i * M:(i + 1) * M, 1:] = m.Vk
                V_table[i * M:(i + 1) * M, 0] = int(i)
        columns = ['modelno'] + ['v' + str(i) for i in range(1, self.kSingularValuesToKeep + 1)]
        vdf = pd.DataFrame(columns=columns, data= V_table)
        vdf['tscolumn'] = (vdf.index - 0.5 * M * vdf['modelno']).astype(int)
        self.writeTable(vdf, tableNames[1], host, database, user, password)

        s_table = np.zeros([len(self.models), 1 + self.kSingularValuesToKeep])
        for i, m in self.models.items():
            s_table[i, 1:] = m.sk
            s_table[i, 0] = int(i)
        columns = ['modelno'] + ['s' + str(i) for i in range(1, self.kSingularValuesToKeep + 1)]
        sdf = pd.DataFrame(columns=columns , data=s_table)
        self.writeTable(sdf, tableNames[2], host, database, user, password)

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
        self.writeTable(cdf, tableNames[3], host, database, user, password)

        conn = psycopg2.connect(host=host, database=database, user=user, password=password)
        cur = conn.cursor()
        #

        c1 = 'CREATE INDEX ON '+tableNames[0]+' (tsrow);'
        c2 = 'CREATE INDEX ON '+tableNames[1]+' (tscolumn);'
        c3 = 'CREATE INDEX ON '+tableNames[2]+' (modelno);'
        c4 = 'CREATE INDEX ON '+tableNames[3]+' (modelno);'
        c5 = 'CREATE INDEX ON '+tableNames[3]+' (coeffpos);'
        cur.execute(c1)
        cur.execute(c2)
        cur.execute(c3)
        cur.execute(c4)
        cur.execute(c5)
        conn.commit()
        cur.close()
        conn.close()

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

    def predict(self, index=None, method='average', NoModels=None, dataPoints = None):
        if NoModels is None: NoModels = len(self.models)
        # if index next predict

        if dataPoints is None and (index is None or index == self.TimeSeriesIndex + 1):
            TSDF = pd.DataFrame(data={'t1': self.TimeSeries[-self.L:]})

            UsedModels = [a[1] for a in self.models.items()[-NoModels:]]
            predicions = np.array([mod.predict(pd.DataFrame(data={}), TSDF) for mod in UsedModels])
            return np.mean(predicions)

        elif dataPoints is None and index <= self.TimeSeriesIndex:
            slack = self.TimeSeriesIndex - index + 1
            if slack > (self.T-self.L): raise Exception
            TSDF  = pd.DataFrame(data={'t1': self.TimeSeries[-self.L-slack:-slack]})
            UsedModels = [a[1] for a in self.models.items()[-NoModels-1:-1]]
            predicions = np.array([mod.predict(pd.DataFrame(data={}), TSDF) for mod in UsedModels])
            return np.mean(predicions)

        elif dataPoints is not None:
            TSDF  = pd.DataFrame(data={'t1': dataPoints})
            UsedModels = [a[1] for a in self.models.items()[-NoModels:]]
            predicions = np.array([mod.predict(pd.DataFrame(data={}), TSDF) for mod in UsedModels])
            return np.mean(predicions)
        else:
            return 0
            # if not predict till then
            # get models weight and average all predictions

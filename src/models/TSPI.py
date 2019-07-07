import numpy as np
import pandas as pd
from  tslib.src.models.tsSVDModel import SVDModel
from math import ceil
from tslib.src.database.DBInterface import Interface
from  tslib.src.models.TSModel import TSmodel
from time import time 

class TSPI(object):
    # kSingularValuesToKeep:    (int) the number of singular values to retain
    # T0:                       (int) the number of entries below which the model will not be trained.
    # T:                        (int) Number of entries in each submodel
    # gamma:                    (float) (0,1) fraction of T after which the model is updated
    # rectFactor:               (int) the ration of no. columns to the number of rows in each sub-model

    def __init__(self, rank = 3,rank_var= 1, T=int(1e5),T_var = None, gamma=0.2, T0=1000, rectFactor=10, interface=Interface
                 , time_series_table = ['basic_ts','ts','row_id'], model_table_name='test', SSVT = False, p = 1.0, direct_var = True, L = None):

        self.k = rank
        self.T = int(T)
        if T_var is None:
            self.T_var = 10*self.T
        else:
            self.T_var = T_var
        self.gamma = gamma
        self.T0 = T0
        self.db_interface = interface
        self.model_tables_name = model_table_name
        self.time_series_table = time_series_table
        self.k_var = rank_var
        self.ts_model = TSmodel(self.k, self.T, self.gamma, self.T0, col_to_row_ratio=rectFactor, interface=self.db_interface, model_table_name = self.model_tables_name, SSVT = False, p = p, L = L)
        self.var_model = TSmodel(self.k_var, self.T_var, self.gamma, self.T0, col_to_row_ratio=rectFactor, interface=self.db_interface,  model_table_name = self.model_tables_name+'_variance', SSVT = SSVT, p =p, L = L)
        self.direct_var = direct_var

    def create_index(self):
        pass

    def update_index(self):
        """
        This function query new datapoints from the database using the variable self.TimeSeriesIndex and call the
        update_model function
        """
        new_entries = self.get_range(self.ts_model.TimeSeriesIndex)
        print 'new data points are obtianed'
        if len(new_entries)>0:
            self.update_model(new_entries)
            self.write_model()



    def update_model(self, NewEntries):
        """
        This function takes a new set of entries and update the model accordingly.
        if the number of new entries means new model need to be bulit, this function segment the new entries into
        several entries and then feed them to the update_ts and fit function
        :param NewEntries: Entries to be included in the new model
        """
        # Define update chunck for the update SVD function (Not really needed, should be resolved once the update function is fixed)
        self.ts_model.update_model(np.array(NewEntries))
        # means = self.ts_model.get_range_predictions(self.var_model.TimeSeriesIndex, self.ts_model.TimeSeriesIndex)
        if self.k_var:
            if self.direct_var:
                means = self.ts_model.denoiseTS()[self.var_model.TimeSeriesIndex:self.ts_model.TimeSeriesIndex]
                obs = np.array(NewEntries)
                var_entries = np.square(obs[:len(means)]-means)
                var_entries[np.isnan(obs[:len(means)])] = 0#np.median(var_entries[np.logical_not(np.isnan(obs[:len(means)]))])
                self.var_model.update_model(var_entries)
            else:
                obs = np.array(NewEntries)
                var_entries = np.square(obs[:])
                self.var_model.update_model(var_entries)


    def write_model(self):
        """
        -
        """
        self.ts_model.WriteModel()
        self.var_model.WriteModel()


    def get_prediction_range(self, t1, t2, model = 'means'):
        """
        call get_imp or get_fore depending on t
        """
        if model =='var':
            model = self.var_model
        else:
            model = self.ts_model
        if t1 > (model.MUpdateIndex-1):
            return self.get_forecast_range(t1,t2, model)
        elif t2 <= (model.MUpdateIndex-1):
            return self.get_imputation_range(t1,t2, model)
        else: return list(self.get_imputation_range(t1,model.MUpdateIndex-1, model))+ list(self.get_forecast_range(model.MUpdateIndex,t2, model))

    def get_forecast_range(self, t1, t2, model):
        connection = self.db_interface.engine.connect()
        coeffs = np.array(self.db_interface.get_coeff(model.model_tables_name + '_c_view', 'average', connection))
        no_coeff = len(coeffs)
        output = np.zeros([t2 - t1 + 1 + no_coeff])
        # output[:no_coeff] = TSPD.db_interface.get_time_series(TSPD.time_series_table[0], t1 - no_coeff, t1 - 1,
        #                                                       value_column=TSPD.time_series_table[1],
        #                                                       index_col=TSPD.time_series_table[2], Desc=False)[:, 0]
        output[:no_coeff] = self.get_imputation_range( t1 - no_coeff, t1-1, model)

        for i in range(0, t2 + 1 - t1):
            output[i + no_coeff] = np.dot(coeffs.T, output[i:i + no_coeff])
            # output[i + no_coeff] = sum([a[0]*b for a, b in zip(coeffs,output[i:i + no_coeff])])
        return output[-(t2 - t1 + 1):]

    def get_imputation_range(self, t1, t2, model):
        connection = self.db_interface.engine.connect()
        m1 = model.get_model_index(t1)
        m2 = model.get_model_index(t2)
        N1 = model.models[m1].N
        N2 = model.models[m2].N
        M1 = model.T/model.L
        if m2 == len(model.models)-1:
            tscol2 =  (t2-model.models[m2].start)/ N2 + (model.models[m2].start)/ model.L
            tsrow2 = (t2-model.models[m2].start) % N2
        else:
            tscol2 = t2 / N2
            tsrow2 = t2 % N2

        if m1 == len(model.models) - 1:
            tscol1 = (t1 - model.models[m1].start) / N1 + (model.models[m1].start) / model.L
            tsrow1 = (t1-model.models[m1].start) % N1
        else:
            tscol1 = t1 / N1
            tsrow1 = t1 % N1

        i_index = (t1 - t1 % N1)
        last_model = len(model.models) - 1
        name = model.model_tables_name
        if tscol1 == tscol2:

            S = model.db_interface.get_S_row(name+'_s', [m1, m2 + 1], model.kSingularValuesToKeep, connection,return_modelno=True)
            U = model.db_interface.get_U_row(name+'_u', [tsrow1, tsrow2], [m1, m2 + 1], model.kSingularValuesToKeep,connection,
                                            return_modelno=True)
            V = model.db_interface.get_V_row(name+'_v', [tscol1, tscol2], model.kSingularValuesToKeep,connection, [m1, m2 + 1],
                                            return_modelno=True)
            p = np.dot(U[U[:, 0] == m1, 1:] * S[0, 1:], V[V[:, 0] == m1, 1:].T)
            if (m2 != last_model and m1 != 0):
                Result = 0.5 * p.T.flatten() + 0.5 * np.dot(U[U[:, 0] == m1 + 1, 1:] * S[1, 1:],
                                                            V[V[:, 0] == m1 + 1, 1:].T).T.flatten()
            else:
                Result = p.T.flatten()
            return Result

        else:

            Result = np.zeros([(t2 + N2 - t2 % N2) - i_index])
            S = model.db_interface.get_S_row(name+'_s', [m1, m2 + 1], model.kSingularValuesToKeep,connection, return_modelno=True)
            U = model.db_interface.get_U_row(name+'_u', [0, model.L], [m1, m2 + 1], model.kSingularValuesToKeep, connection,
                                            return_modelno=True)
            V = model.db_interface.get_V_row(name+'_v', [tscol1, tscol2], model.kSingularValuesToKeep,connection, [m1, m2 + 1],
                                            return_modelno=True)
            for m in range(m1, m2 + 1 + (m2 != last_model)):
                tscol_i = max(tscol1, m * M1 / 2)
                tscol_f = min(tscol2, (m + 2) * M1 / 2 - 1)
                length = N1 * (tscol_f - tscol_i + 1)
                i = tscol_i * N1 - i_index
                p = np.dot(U[U[:, 0] == m, 1:] * S[m - m1, 1:], V[V[:, 0] == m, 1:].T)
                Result[i:i + length] += 0.5 * p.T.flatten()
            Result[:model.T / 2] = (1 + (m1 == 0)) * Result[:model.T / 2]
            end = -N2 + tsrow2 + 1
            if end == 0: end = None
            return Result[tsrow1:end]

    #

    def get_prediction(self, t, model = 'means'):
        """
        call get_imp or get_fore depending on t
        """
        if self.direct_var or model == 'means':
            if model =='var':
                model = self.var_model
            else:
                model = self.ts_model
            if t > (model.MUpdateIndex-1):
                return self.get_forecast(t, model)
            else:
                return self.get_imputation(t, model)
        else:
            if t > (model.MUpdateIndex-1):
                return self.get_forecast(t, self.var_model) - (self.get_forecast(t,  self.ts_model))**2
            else:
                return self.get_imputation(t, self.var_model) - (self.get_imputation(t, self.ts_model)) ** 2


    def get_imputation(self, t, model):
        """
        implement the same singles point query. use get from table function in interface
        """
        connection = self.db_interface.engine.connect()
        modelNo = model.get_model_index(t+1)
        N = model.models[modelNo].N
        tscolumn = t/ N
        tsrow = t%N

        if modelNo == len(model.models)-1:
            tscolumn = (t - model.models[modelNo].start) / N + (model.models[modelNo].start) / model.L
            tsrow = (t - model.models[modelNo].start)%N
        U,V,S = self.db_interface.get_SUV(model.model_tables_name, [tscolumn, tscolumn], [tsrow,tsrow], [modelNo,modelNo+1], model.kSingularValuesToKeep, connection )

        if len(S[0]) == 2 and len(V[0]) ==2 and len(U[0]) ==2:
            return 0.5*(sum([a[0]*b[0]*c[0] for a, b,c in zip(U,S,V)]) + sum([a[1]*b[1]*c[1] for a, b,c in zip(U,S,V)]))
   
        return sum([a[0]*b[0]*c[0] for a, b,c in zip(U,S,V)])
        # U = self.db_interface.get_U_row(model.model_tables_name+'_u', [tsrow,tsrow], [modelNo,modelNo+1],model.kSingularValuesToKeep, connection)
        # V = self.db_interface.get_V_row(model.model_tables_name + '_v', [tscolumn, tscolumn], model.kSingularValuesToKeep, connection)
        # S = self.db_interface.get_S_row(model.model_tables_name + '_s', [modelNo,modelNo+1], model.kSingularValuesToKeep, connection)
        
        # U1 = U[0, :]
        # V1 = V[0, :]
        # S1 = S[0, :]

        # if len(S) == 2 and len(V) ==2 and len(U) ==2:
        #     U2 = U[1, :]
        #     V2 = V[1, :]
        #     S2 = S[1, :]
        #     return 0.5*(np.dot(U1*S1,V1.T) + np.dot(U2*S2,V2.T))
   
        # return np.dot(U1*S1,V1.T)



    def get_forecast(self, t, model):
        """
        """
        connection = self.db_interface.engine.connect()
        t1 = time()
        coeffs = self.db_interface.get_coeff( model.model_tables_name+'_c_view', 'average', connection)
        no_coeff = len(coeffs)
        last_obs = self.db_interface.get_time_series(self.time_series_table[0],t-no_coeff, t-1,connection, value_column=self.time_series_table[1], index_col=self.time_series_table[2], Desc = False)
      
        return sum([a[0]*b[0] for a, b in zip(coeffs,last_obs)])
        # return np.multiply(coeffs, last_obs)


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

        return np.array(self.db_interface.get_time_series(self.time_series_table[0], t1, t2,
                                                 value_column=self.time_series_table[1],
                                                 index_col=self.time_series_table[2]))[:,0]


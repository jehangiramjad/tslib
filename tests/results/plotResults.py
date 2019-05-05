import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Data = pd.read_csv('Results_harmonicsTrend_vs_ModelSize_patch_8_0.9.csv')
ModelSize = Data['ModelSize']
ImpMean = Data['ImpErrMean']
ImpObs = Data['ImpErrObs']
ForeMean = Data['ForErrMean']
ForeObs = Data['ForErrObs']
Timing = Data['timing']

plt.figure()
plt.plot(ModelSize,ImpMean, label = 'error of Imputation vs. mean')
plt.plot(ModelSize,ImpObs,'--', label = 'error of Imputation vs. obs')
plt.xlabel('Number of rows in each sub model (L)')
plt.ylabel('RMSE')
plt.legend()

plt.figure()
plt.plot(ModelSize,ForeMean, label = 'error of Forecasting vs. mean')
plt.plot(ModelSize,ForeObs,'--', label = 'error of Forecasting vs. obs')
plt.xlabel('Number of rows in each sub model (L)')
plt.ylabel('RMSE')
plt.legend()

plt.figure()
plt.plot(ModelSize,1e6*Timing, label = 'error of Imputation vs. mean')

plt.xlabel('Number of rows in each sub model (L)')
plt.ylabel('Update time per data point (micro seconds)')

plt.show()
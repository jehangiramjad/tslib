import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Data = pd.read_csv('timingQueriesC.csv')
q1 = Data['ModelQuery']
q2 = Data['DBQuery']
q3 = Data['ForecastQuery']
lq2 = q2[q2<np.mean(q2)+50*np.sqrt(np.var(q2))]
lq1 = q1[q1<np.mean(q1)+50*np.sqrt(np.var(q1))]
lq3 = q3[q3<np.mean(q3)+50*np.sqrt(np.var(q3))]
plt.hist(lq1, bins=np.arange(0, 0.001, 0.000002), ls='dashed',  fc=(1, 0.2, 0, 0.3), label = 'Model query (impute), median = %.2f  ms '% (1000*np.median(lq1)))
plt.hist(lq2, bins=np.arange(0, 0.001, 0.000002), ls='dotted',  fc=(0, 0, 0, 0.3),label ='DB query, median = %.2f ms '%(1000*np.median(lq2)))
plt.hist(lq3, bins=np.arange(0.0002, 0.003, 0.000002), ls='dotted',  fc=(0.2, 0.3, 1, 0.3),label ='Model query(forecast), median = %.2f ms '%(1000*np.median(lq3)))
plt.legend()
plt.show()
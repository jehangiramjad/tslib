import sys, os
sys.path.append("../../..")
sys.path.append("..")
sys.path.append(os.getcwd())

import pandas as pd
from fbprophet import Prophet

import numpy as np

import matplotlib.pyplot as plt

import tslib.src.tsUtils as tsUtils


for name in ['harmonic', 'trend', 'arma']:#['harmtrend', 'negexp', 'harmvar']:

	trainDF = pd.read_pickle('./'+name+'/trainDF.pkl')
	testDF = pd.read_pickle('./'+name+'/testDF.pkl')

	dictn={'ds': trainDF.index, 'y':trainDF['t1']}

	print(len(trainDF), len(testDF))

	df = pd.DataFrame(data=dictn)

	m = Prophet()
	m.fit(df)

	future = m.make_future_dataframe(periods=len(testDF))

	forecast = m.predict(future)

	# fig1=m.plot(forecast)

	# fig1.savefig('./'+name+'/prophet.png')

	# fig2 = m.plot_components(forecast)


	# xs = np.arange(len(forecast))

	# plt.plot(xs, forecast['yhat'].tolist())
	# plt.show()


	print(" RMSE (prediction vs obs)  = %f" %tsUtils.rmse(testDF['t1'].values, forecast['yhat'].tolist()[-len(testDF):]))



# df = pd.read_csv('example_wp_log_peyton_manning.csv')[:-5]
# df2 = pd.read_pickle('./'+'peyton'+'/trainMasterDF.pkl')['t1'].values
# df3 = pd.read_pickle('./'+'peyton'+'/testDF.pkl')['t1'].values

# df['y']=np.concatenate([df2,df3], axis = 0)

# print(df.head())

# trainDF = df[:2600]
# testDF = df[2600:]
# print(len(trainDF), len(testDF))

# m = Prophet()
# m.fit(trainDF)

# future = m.make_future_dataframe(periods = len(testDF))

# forecast = m.predict(future)

# fig1=m.plot(forecast)

# fig1.savefig('./peyton/peyton_proph.png')

# print(" RMSE (prediction vs obs)  = %f" %tsUtils.rmse(df3, forecast['yhat'].tolist()[-len(df3):]))

import pandas as pd
from fbprophet import Prophet

for name in ['harmtrend', 'negexp', 'harmvar']:
	nbrSingValuesToKeep=5
	pObservation=1.0
	trainMasterDF = pd.read_pickle('./'+name+'/trainMasterDF.pkl')
	trainDF = pd.read_pickle('./'+name+'/trainDF.pkl')
	meanTrainDF = pd.read_pickle('./'+name+'/meanTrainDF.pkl')
	testDF = pd.read_pickle('./'+name+'/testDF.pkl')
	meanTestDF = pd.read_pickle('./'+name+'/meanTestDF.pkl')

	trainDF['ds'] = trainDF.index

	trainDF.columns=['ds', 'y']

	m = Prophet()
	m.fit(trainDF)

	future = m.make_future_dataframe(periods=len(testDF), freq = 'T', include_history = False)
	print(future.head(), future.tail())

	forecast = m.predict(future)
	fig1 = m.plot(forecast)

	# print(" RMSE (prediction vs obs)  = %f" %tsUtils.rmse(testDF[key1].values, forecastArray))

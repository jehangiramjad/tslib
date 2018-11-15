import sys, os
sys.path.append("../../..")
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
from  tslib.src.models.tsALSModel import ALSModel
import tslib.src.tsUtils as tsUtils

def make_df(combinedTS, meanTS, N, M, p, name):
	timeSteps = N*M

	# train/test split
	trainProp = 0.9
	M1 = int(trainProp * M)
	M2 = M - M1

	trainPoints = N*M1
	testPoints = N*M2

	max1 = np.nanmax(combinedTS)
	min1 = np.nanmin(combinedTS)
	max2 = np.nanmax(meanTS)
	min2 = np.nanmin(meanTS)
	max = np.max([max1, max2])
	min = np.min([min1, min2])

	combinedTS = tsUtils.normalize(combinedTS, max, min)
	meanTS = tsUtils.normalize(meanTS, max, min)

	# produce timestamps
	timestamps = np.arange('1900-01-01', timeSteps, dtype='datetime64[D]') # arbitrary start date

	# split the data
	trainDataMaster = combinedTS[0:trainPoints] # need this as the true realized values for comparisons later
	meanTrainData = meanTS[0:trainPoints] # this is only needed for various statistical comparisons later

	# randomly hide training data: choose between randomly hiding entries or randomly hiding consecutive entries
	(trainData, pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(trainDataMaster), p)

	# now further hide consecutive entries for a very small fraction of entries in the eventual training matrix
	(trainData, pObservation) = tsUtils.randomlyHideConsecutiveEntries(copy.deepcopy(trainData), 0.9, int(M1 * 0.25), M1)

	# interpolating Nans with linear interpolation
	#trainData = tsUtils.nanInterpolateHelper(trainData)

	# test data and hidden truth
	testData = combinedTS[-1*testPoints: ]
	meanTestData = meanTS[-1*testPoints: ] # this is only needed for various statistical comparisons

	# time stamps
	trainTimestamps = timestamps[0:trainPoints]
	testTimestamps = timestamps[-1*testPoints: ]

	# once we have interpolated, pObservation should be set back to 1.0
	pObservation = 1.0

	# create pandas df
	key1 = 't1'
	a = pd.to_datetime(trainTimestamps, errors = 'ignore')
	trainMasterDF = pd.DataFrame(index=a, data={key1: trainDataMaster}) # needed for reference later
	trainDF = pd.DataFrame(index=a, data={key1: trainData})
	meanTrainDF = pd.DataFrame(index=a, data={key1: meanTrainData})

	b = pd.to_datetime(testTimestamps, errors = 'ignore')
	testDF = pd.DataFrame(index=b, data={key1: testData})
	meanTestDF = pd.DataFrame(index=b, data={key1: meanTestData})

	# if not os.path.isdir("name"):
	# 	os.mkdir(name)

	# trainMasterDF.to_pickle('./'+name+'/trainMasterDF.pkl')
	# trainDF.to_pickle('./'+name+'/trainDF.pkl')
	# meanTrainDF.to_pickle('./'+name+'/meanTrainDF.pkl')
	# testDF.to_pickle('./'+name+'/testDF.pkl')
	# meanTestDF.to_pickle('./'+name+'/meanTestDF.pkl')

	return (trainMasterDF, trainDF, testDF)

def loaded_test(trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF, N, M, nbrSingValuesToKeep, pObservation, means = True):

	# mod = SVDModel(key1, nbrSingValuesToKeep, N, M1, probObservation=pObservation, svdMethod='numpy', otherSeriesKeysArray=[], includePastDataOnly=True)
	# mod.fit(trainDF)
	# imputedDf = mod.denoisedDF()

	# print(" RMSE (training imputation vs mean) = %f" %tsUtils.rmse(meanTrainDF[key1].values, imputedDf[key1].values))
	# print(" RMSE (training imputation vs obs)  = %f" %tsUtils.rmse(trainMasterDF[key1].values, imputedDf[key1].values))

	imputedDf=trainDF

	p=1.0

	key1 = 't1'

	trainProp = 0.9
	M1 = int(trainProp * M)

	mod = SVDModel(key1, nbrSingValuesToKeep, N, M1, probObservation=pObservation, svdMethod='numpy', otherSeriesKeysArray=[], includePastDataOnly=True)
	mod.fit(trainDF)

	print("Forecasting (#points = %d)..." %len(testDF))
	# test data is used for point-predictions
	forecastArray = []
	for i in range(0, len(testDF)):
	    pastPoints = np.zeros(N-1) # need an N-1 length vector of past point
	    j = 0
	    if (i < N - 1):   # the first prediction uses the end of the training data
	        while (j < N - 1 - i):
	            pastPoints[j] = trainMasterDF[key1].values[len(trainDF) - (N - 1 - i) + j]
	            j += 1

	    if (j < N - 1): # use the new test data
	        pastPoints[j:] = testDF[key1].values[i - (N - 1) + j:i] 

	    keyToSeriesDFNew = pd.DataFrame(data={key1: pastPoints})
	    prediction = mod.predict(pd.DataFrame(data={}), keyToSeriesDFNew, bypassChecks=False)
	    forecastArray.append(prediction)

	if means: print(" RMSE (prediction vs mean) = %f" %tsUtils.rmse(meanTestDF[key1].values, forecastArray))
	print(" RMSE (prediction vs obs)  = %f" %tsUtils.rmse(testDF[key1].values, forecastArray))

	print("Plotting...")
	plt.plot(np.concatenate((trainMasterDF[key1].values, testDF[key1].values), axis=0), color='gray', label='Observed')
	if means: plt.plot(np.concatenate((meanTrainDF[key1].values, meanTestDF[key1].values), axis=0), color='red', label='True Means')
	plt.plot(np.concatenate((imputedDf[key1].values, forecastArray), axis=0), color='blue', label='Forecasts')
	plt.axvline(x=len(trainDF), linewidth=1, color='black', label='Training End')
	legend = plt.legend(loc='upper left', shadow=True)
	# plt.title('Single Time Series (ARMA + Periodic + Trend) - $p = %.2f$' %p)
	plt.show()

def csv_test(filename, N, M, takeoff, name):
	fullDF = pd.read_csv(filename)
	combinedTS = np.array(fullDF[fullDF.columns[1]])[:-takeoff]
	print(len(combinedTS))

	(trainMasterDF, trainDF, testDF) = make_df(combinedTS, combinedTS, N=N, M=M, p=1.0, name=name)

	loaded_test(trainMasterDF, trainDF, trainDF, testDF, testDF, N, M, 5, 1.0, False)





def test(combinedTS, meanTS, N, M, p=1.0):
	
	timeSteps=N*M
	# train/test split
	trainProp = 0.9
	M1 = int(trainProp * M)
	M2 = M - M1

	trainPoints = N*M1
	testPoints = N*M2

	# train/test split
	trainProp = 0.9
	M1 = int(trainProp * M)
	M2 = M - M1

	trainPoints = N*M1
	testPoints = N*M2

	# produce timestamps
	timestamps = np.arange('2017-09-10 20:30:00', timeSteps, dtype='datetime64[1m]') # arbitrary start date

	# split the data
	trainDataMaster = combinedTS[0:trainPoints] # need this as the true realized values for comparisons later
	meanTrainData = meanTS[0:trainPoints] # this is only needed for various statistical comparisons later

	# randomly hide training data: choose between randomly hiding entries or randomly hiding consecutive entries
	(trainData, pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(trainDataMaster), p)

	# now further hide consecutive entries for a very small fraction of entries in the eventual training matrix
	# (trainData, pObservation) = tsUtils.randomlyHideConsecutiveEntries(copy.deepcopy(trainData), 0.9, int(M1 * 0.25), M1)

	# interpolating Nans with linear interpolation
	#trainData = tsUtils.nanInterpolateHelper(trainData)

	# test data and hidden truth
	testData = combinedTS[-1*testPoints: ]
	meanTestData = meanTS[-1*testPoints: ] # this is only needed for various statistical comparisons

	# time stamps
	trainTimestamps = timestamps[0:trainPoints]
	testTimestamps = timestamps[-1*testPoints: ]

	# once we have interpolated, pObservation should be set back to 1.0
	pObservation = 1.0

	# create pandas df
	key1 = 't1'
	trainMasterDF = pd.DataFrame(index=trainTimestamps, data={key1: trainDataMaster}) # needed for reference later
	trainDF = pd.DataFrame(index=trainTimestamps, data={key1: trainData})
	meanTrainDF = pd.DataFrame(index=trainTimestamps, data={key1: meanTrainData})

	testDF = pd.DataFrame(index=testTimestamps, data={key1: testData})
	meanTestDF = pd.DataFrame(index=testTimestamps, data={key1: meanTestData})

	nbrSingValuesToKeep=5

	mod = SVDModel(key1, nbrSingValuesToKeep, N, M1, probObservation=pObservation, svdMethod='numpy', otherSeriesKeysArray=[], includePastDataOnly=True)
	mod.fit(trainDF)
	imputedDf = mod.denoisedDF()

	print(" RMSE (training imputation vs mean) = %f" %tsUtils.rmse(meanTrainDF[key1].values, imputedDf[key1].values))
	print(" RMSE (training imputation vs obs)  = %f" %tsUtils.rmse(trainMasterDF[key1].values, imputedDf[key1].values))

	print("Forecasting (#points = %d)..." %len(testDF))
	# test data is used for point-predictions
	forecastArray = []
	for i in range(0, len(testDF)):
	    pastPoints = np.zeros(N-1) # need an N-1 length vector of past point
	    j = 0
	    if (i < N - 1):   # the first prediction uses the end of the training data
	        while (j < N - 1 - i):
	            pastPoints[j] = trainMasterDF[key1].values[len(trainDF) - (N - 1 - i) + j]
	            j += 1

	    if (j < N - 1): # use the new test data
	        pastPoints[j:] = testDF[key1].values[i - (N - 1) + j:i] 

	    keyToSeriesDFNew = pd.DataFrame(data={key1: pastPoints})
	    prediction = mod.predict(pd.DataFrame(data={}), keyToSeriesDFNew, bypassChecks=False)
	    forecastArray.append(prediction)

	print(" RMSE (prediction vs mean) = %f" %tsUtils.rmse(meanTestDF[key1].values, forecastArray))
	print(" RMSE (prediction vs obs)  = %f" %tsUtils.rmse(testDF[key1].values, forecastArray))

	print("Plotting...")
	plt.plot(np.concatenate((trainMasterDF[key1].values, testDF[key1].values), axis=0), color='gray', label='Observed')
	plt.plot(np.concatenate((meanTrainDF[key1].values, meanTestDF[key1].values), axis=0), color='red', label='True Means')
	plt.plot(np.concatenate((imputedDf[key1].values, forecastArray), axis=0), color='blue', label='Forecasts')
	plt.axvline(x=len(trainDF), linewidth=1, color='black', label='Training End')
	legend = plt.legend(loc='upper left', shadow=True)
	plt.title('Single Time Series (ARMA + Periodic + Trend) - $p = %.2f$' %p)
	plt.show()

# for obs, mean, name in [('armaobs.npy', 'armamean.npy', 'arma'), ('harmonic.npy', 'harmonic.npy', 'harmonic'), ('trend.npy', 'trend.npy', 'trend')]:#('../harmobs.npy', '../harmmean.npy', 'harmtrend'),('../negexpsdobs.npy', '../negexpsdmean.npy', 'negexp'), ('../sdharmonicobs.npy','../sdharmonicmean.npy','harmvar')]:
# 	obsts = np.load(obs)
# 	meants = np.load(mean)
# 	make_df(obsts, meants, 100, 1000, 1.0, name)

# for name in ['arma', 'harmonic', 'trend']:
# 	nbrSingValuesToKeep=5
# 	pObservation=1.0
# 	trainMasterDF = pd.read_pickle('./'+name+'/trainMasterDF.pkl')
# 	trainDF = pd.read_pickle('./'+name+'/trainDF.pkl')
# 	meanTrainDF = pd.read_pickle('./'+name+'/meanTrainDF.pkl')
# 	testDF = pd.read_pickle('./'+name+'/testDF.pkl')
# 	meanTestDF = pd.read_pickle('./'+name+'/meanTestDF.pkl')

# 	loaded_test(trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF, 100, 1000, nbrSingValuesToKeep, pObservation)

# csv_test(filename='example_wp_log_peyton_manning.csv', N=50, M=58, takeoff=5, name='peyton')
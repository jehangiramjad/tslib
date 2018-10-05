import sys, os
sys.path.append("../..")
sys.path.append("..")
sys.path.append(os.getcwd())

#from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy

from tslib.src.data import generateHarmonics as gH
from  tslib.src.data import generateTrend as gT
import tslib.src.data.generateARMA as gA
from  tslib.src.models.tsSVDModel import SVDModel
from  tslib.src.models.tsALSModel import ALSModel
import tslib.src.tsUtils as tsUtils

#############################################################
# TODOS:
# Lots of small harmonics
# Seasonal Data - feed in old array as init array for arma
# Changing variance
# Changing variance over time
# change num singular vals
# ([-1.0, 1.337, -0.715, 0.191, -0.026, 0.001], [-1.0, 3.054, -3.731, 2.279, -0.696, 0.085])
#############################################################




#############################################################
# Generating Data
#############################################################

def armaData(arLags, maLags, noiseSD, timeSteps, startingArray=None):
	if startingArray==None: 
		startingArray = np.zeros(np.max([len(arLags), len(maLags)])) # start with all 0's
	noiseMean = 0.0

	(observedArray, meanArray, errorArray) = gA.generate(arLags, maLags, startingArray, timeSteps, noiseMean, noiseSD)

	return (observedArray, meanArray)

def trendData(trends_fns, timeSteps): #trend_fns should be list of tuples (trend_function, power if linear or dampenign otherwise, displancement) 
	data=np.zeros(timeSteps)
	for fn, power_or_dampening, displacement in trends_fns:
		data += gT.generate(fn, power_or_dampening, displacement, timeSteps)
	return data

def harmonicData(sineCoeffs, sinePeriods, cosineCoeffs, cosine_periods):
	data = gH.generate(sineCoeffs, sinePeriods, cosineCoeffs, cosinePeriods, timeSteps)
	return data

def generateStationaryARMALags(arOrder, maOrder):
	arPolyCoeffs = np.poly1d([1.0])
	invRoot=0.0
	for i in range(arOrder):
		while invRoot==0.0: 
			invRoot=np.random.sample()
		arPolyCoeffs=arPolyCoeffs*np.poly1d([invRoot, -1])

	maPolyCoeffs = np.poly1d([1.0])
	invRoot=0.0
	for i in range(maOrder):
		while invRoot==0.0: 
			invRoot=np.random.sample()
		maPolyCoeffs=maPolyCoeffs*np.poly1d([invRoot, -1])

	arLags = np.around(arPolyCoeffs,3).tolist()
	arLags.reverse()

	maLags = np.around(maPolyCoeffs,3).tolist()
	maLags.reverse()

	if arOrder==0: arLags=[]
	if maOrder==0: maLags=[]

	return (arLags, maLags)

#############################################################
# Formatting Data
#############################################################

def normalize(combinedTS, meanTS):
	
	#normalize the values to all lie within [-1, 1] -- helps with RMSE comparisons
    # can use the tsUtils.unnormalize() function to convert everything back to the original range at the end, if needed
    max1 = np.nanmax(combinedTS)
    min1 = np.nanmin(combinedTS)
    max2 = np.nanmax(meanTS)
    min2 = np.nanmin(meanTS)
    max = np.max([max1, max2])
    min = np.min([min1, min2])

    combinedTS = tsUtils.normalize(combinedTS, max, min)
    meanTS = tsUtils.normalize(meanTS, max, min)

    return (combinedTS, meanTS)

def makeDFs(combinedTS, meanTS, N=500, M=400, p=.7):
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
	trainMasterDF = pd.DataFrame(index=trainTimestamps, data={key1: trainDataMaster}) # needed for reference later
	trainDF = pd.DataFrame(index=trainTimestamps, data={key1: trainData})
	meanTrainDF = pd.DataFrame(index=trainTimestamps, data={key1: meanTrainData})

	testDF = pd.DataFrame(index=testTimestamps, data={key1: testData})
	meanTestDF = pd.DataFrame(index=testTimestamps, data={key1: meanTestData})

	return (trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF)

#############################################################
# Plotting Series
#############################################################

def plotSeries(combinedTS, title=''):
	plt.title(title)
	plt.plot(np.arange(combinedTS.shape[0]), combinedTS)
	plt.show()

#############################################################
# Test Function
#############################################################

def test(combinedTS, meanTS, nbrSingValuesToKeep=5, N=50, M=400, p=.7, SVD=True, ALS=True):
	# #prob redundant
	# timeSteps=N*M
	# # train/test split
	# trainProp = 0.9
	# M1 = int(trainProp * M)
	# M2 = M - M1

	# trainPoints = N*M1
	# testPoints = N*M2

	# # train/test split
	# trainProp = 0.9
	# M1 = int(trainProp * M)
	# M2 = M - M1

	# trainPoints = N*M1
	# testPoints = N*M2
	# #end reduncy
	# key1='t1'
	# trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF = makeDFs(combinedTS, meanTS, N, M, p)
	# timeSteps=N*M

	out=[]









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
	trainMasterDF = pd.DataFrame(index=trainTimestamps, data={key1: trainDataMaster}) # needed for reference later
	trainDF = pd.DataFrame(index=trainTimestamps, data={key1: trainData})
	meanTrainDF = pd.DataFrame(index=trainTimestamps, data={key1: meanTrainData})

	testDF = pd.DataFrame(index=testTimestamps, data={key1: testData})
	meanTestDF = pd.DataFrame(index=testTimestamps, data={key1: meanTestData})









	if SVD:
		# train the model
		mod = SVDModel(key1, nbrSingValuesToKeep, N, M1, probObservation=pObservation, svdMethod='numpy', otherSeriesKeysArray=[], includePastDataOnly=True)
		mod.fit(trainDF)
		imputedDf = mod.denoisedDF()

		svd_imp_v_mean_rmse=tsUtils.rmse(meanTrainDF[key1].values, imputedDf[key1].values)
		svd_imp_v_obs_rmse=tsUtils.rmse(trainMasterDF[key1].values, imputedDf[key1].values)

		out.append(svd_imp_v_mean_rmse)
		out.append(svd_imp_v_obs_rmse)

		print("      SVD RMSE (training imputation vs mean) = %f" %tsUtils.rmse(meanTrainDF[key1].values, imputedDf[key1].values))
		print("      SVD RMSE (training imputation vs obs)  = %f" %tsUtils.rmse(trainMasterDF[key1].values, imputedDf[key1].values))

	if ALS:
		# uncomment below to run the ALS algorithm ; comment out the above line
		mod = ALSModel(key1, nbrSingValuesToKeep, N, M1, probObservation=pObservation, otherSeriesKeysArray=[], includePastDataOnly=True)
		mod.fit(trainDF)

		# imputed + denoised data 
		imputedDf = mod.denoisedDF()

		print("      ALS RMSE (training imputation vs mean) = %f" %tsUtils.rmse(meanTrainDF[key1].values, imputedDf[key1].values))
		print("      ALS RMSE (training imputation vs obs)  = %f" %tsUtils.rmse(trainMasterDF[key1].values, imputedDf[key1].values))

		als_imp_v_mean_rmse=tsUtils.rmse(meanTrainDF[key1].values, imputedDf[key1].values)
		als_imp_v_obs_rmse=tsUtils.rmse(trainMasterDF[key1].values, imputedDf[key1].values)

		out.append(als_imp_v_mean_rmse)
		out.append(als_imp_v_obs_rmse)

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

		als_pred_v_mean=tsUtils.rmse(meanTestDF[key1].values, forecastArray)
		als_pred_v_obs=tsUtils.rmse(testDF[key1].values, forecastArray)

		out.append(als_pred_v_mean)
		out.append(als_pred_v_obs)

		print("      RMSE (prediction vs mean) = %f" %tsUtils.rmse(meanTestDF[key1].values, forecastArray))
		print("      RMSE (prediction vs obs)  = %f" %tsUtils.rmse(testDF[key1].values, forecastArray))

	return out

    # print("Plotting...")
    # plt.plot(np.concatenate((trainMasterDF[key1].values, testDF[key1].values), axis=0), color='gray', label='Observed')
    # plt.plot(np.concatenate((meanTrainDF[key1].values, meanTestDF[key1].values), axis=0), color='red', label='True Means')
    # plt.plot(np.concatenate((imputedDf[key1].values, forecastArray), axis=0), color='blue', label='Forecasts')
    # plt.axvline(x=len(trainDF), linewidth=1, color='black', label='Training End')
    # legend = plt.legend(loc='upper left', shadow=True)
    # plt.title('Single Time Series (ARMA + Periodic + Trend) - $p = %.2f$' %p)
    # plt.show()

#############################################################
# Tests
#############################################################

def StandardDevTest():
	pass

def NumSingularValsTest():
	pass

#ARMA TESTS:
def testARMAStandardDev():
	N, M = 100, 1000
	timeSteps=N*M

	ar, ma = ([-1.0, 1.337, -0.715, 0.191, -0.026, 0.001], [-1.0, 3.054, -3.731, 2.279, -0.696, 0.085])
	
	# trend = trendData([(gT.linearTrendFn, 0.35, -2.5), (gT.logTrendFn, 2.0*float(1.0/N*M), -2.5), (gT.negExpTrendFn, 2.0*float(1.0/N*M), -2.5)], N*M)

	sineCoeffs = [-2.0, 3.0]
	sinePeriods = [26.0, 30.0]

	cosineCoeffs = [-2.5]
	cosinePeriods = [16.0]

	harmonics = gH.generate(sineCoeffs, sinePeriods, cosineCoeffs, cosinePeriods, N*M)

	dampening = 2.0*float(1.0/timeSteps)
	power = 0.35
	displacement = -2.5

	f1 = gT.linearTrendFn
	data = gT.generate(f1, power=power, displacement=displacement, timeSteps=timeSteps)

	f2 = gT.logTrendFn
	data += gT.generate(f2, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

	f3 = gT.negExpTrendFn
	data += gT.generate(f3, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

	trend=data 

	timeSteps=N*M
	data = [[],[],[],[],[],[]]
	

	for i in range(1,11):
		avgs=[0,0,0,0,0,0]
		print 'stdev=',5.0*i
		obs, mean = armaData(ar, ma, 5.0*i, N*M, startingArray=None)

		combinedTS=trend+obs+harmonics
		meanTS=mean+trend+harmonics

		combinedTS, meanTS = normalize(combinedTS, meanTS)

		for i in range(10):
			tests=test(combinedTS, meanTS, nbrSingValuesToKeep=5, N=N, M=M, p=.7, SVD=True, ALS=True)
			
			for i in range(len(tests)):
				avgs[i]+=tests[i]

		for i in range(len(avgs)):
			data[i].append(avgs[i]/10.0)

	data = np.array(data)
	np.save('data.npy', data)
	# xs=5*np.arange(11)[1:]

	# plt.xlabel('Error Stdev')
	# plt.ylabel('RMSE')
	# plt.title('Imputation and Prediction Performance of SVD and ALS on ARMA')
	# plt.plot(xs, data[1], color='g', label='SVD Imputation v. Obv')
	# plt.plot(xs, data[3], color='b', label='ALS Imputation v. Obv')
	# plt.plot(xs, data[5], color='r', label='ALS Prediction v. Obv')
	# plt.legend()
	# plt.show()

		


def testARMANumSingVals():
	N, M = 100, 1000
	timeSteps=N*M

	ar, ma = ([-1.0, 1.337, -0.715, 0.191, -0.026, 0.001], [-1.0, 3.054, -3.731, 2.279, -0.696, 0.085])
	
	# trend = trendData([(gT.linearTrendFn, 0.35, -2.5), (gT.logTrendFn, 2.0*float(1.0/N*M), -2.5), (gT.negExpTrendFn, 2.0*float(1.0/N*M), -2.5)], N*M)

	sineCoeffs = [-2.0, 3.0]
	sinePeriods = [26.0, 30.0]

	cosineCoeffs = [-2.5]
	cosinePeriods = [16.0]

	harmonics = gH.generate(sineCoeffs, sinePeriods, cosineCoeffs, cosinePeriods, N*M)

	dampening = 2.0*float(1.0/timeSteps)
	power = 0.35
	displacement = -2.5

	f1 = gT.linearTrendFn
	data = gT.generate(f1, power=power, displacement=displacement, timeSteps=timeSteps)

	f2 = gT.logTrendFn
	data += gT.generate(f2, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

	f3 = gT.negExpTrendFn
	data += gT.generate(f3, dampening=dampening, displacement=displacement, timeSteps=timeSteps)

	trend=data 

	obs, mean = armaData(ar, ma, 1.0, N*M, startingArray=None)

	combinedTS=trend+obs+harmonics
	meanTS=mean+trend+harmonics

	combinedTS, meanTS = normalize(combinedTS, meanTS)

	timeSteps=N*M
	data = [[],[],[],[],[],[]]
	

	for i in range(1,11):
		avgs=[0,0,0,0,0,0]
		print 'Num Singular Vals',5*i

		for _ in range(10):
			tests=test(combinedTS, meanTS, nbrSingValuesToKeep=5*i, N=N, M=M, p=.7, SVD=True, ALS=True)
			
			for j in range(len(tests)):
				avgs[j]+=tests[j]

		for j in range(len(avgs)):
			data[j].append(avgs[j]/10.0)

	data = np.array(data)
	np.save('data2.npy', data)

#TREND TESTS:
def testTrendStandardDev():
	pass

def testTrendNumSingVals():
	pass

#HARMONIC TESTS:
def testHarmonicStandardDev():
	pass

def testHarmonicNumSingVals():
	pass

#MAIN
def main():
	# a,b = generateStationaryARMALags(5,5)
	# obs, mean = armaData(a, b, 1.0, 200*500, startingArray=None)
	# print((a,b))
	# plotSeries(obs,'f')

	testARMAStandardDev()
	testARMANumSingVals()

	# timeSteps=40*500
	# arLags = [0.268, -1.247, 1.934, -1.0]
	# arLags.reverse()
	# maLags = [0.5, 0.1]

	# startingArray = np.zeros(np.max([len(arLags), len(maLags)])) # start with all 0's
	# noiseMean = 0.0
	# noiseSD = 1.0

	# (observedArray, meanArray, errorArray) = gA.generate(arLags, maLags, startingArray, timeSteps, noiseMean, noiseSD)

	# plotSeries(observedArray)

if __name__ == "__main__":
    main()

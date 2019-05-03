import sys, os
from fbprophet import Prophet

sys.path.append("../../../..")
sys.path.append("..")
sys.path.append(os.getcwd())

#from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy

from tslib.src.data import generateHarmonics as gH
from  tslib.src.data import generateTrend as gT
import tslib.src.data.generateARMA as gA
import tslib.src.data.generateARMAvar as gAv
import tslib.src.data.generateARMAvar as gAvar
from  tslib.src.models.tsSVDModel import SVDModel
from  tslib.src.models.tsALSModel import ALSModel
import tslib.src.tsUtils as tsUtils
import matplotlib.pyplot as plt

from itertools import combinations as combination
from svdpredict import make_df

from svdpredict import make_df_unnorm

def armaData(arLags, maLags, noiseSD, timeSteps, startingArray=None):
	if startingArray==None: 
		startingArray = np.zeros(np.max([len(arLags), len(maLags)])) # start with all 0's
	noiseMean = 0.0

	(observedArray, meanArray, errorArray) = gA.generate(arLags, maLags, startingArray, timeSteps, noiseMean, noiseSD)

	return (observedArray, meanArray)

def generate_singles(timeSteps):
	lags = [([1.0, .25,.2],[]),
	([1.0, .7875,.1125],[]),
	([-1.0, 1.1, -0.484, 0.107, -0.012, 0.001], []),
	([-1.0, 1.526, -0.931, 0.284, -0.043, 0.003], [])]

	names = ['./single/ar25.npy', './single/ar29.npy', './single/ar55.npy', './single/ar59.npy', './single/h10.npy', './single/h3.npy', './single/log.npy', './single/sqrt.npy']

	for i in range(len(lags)):
		ar, ma = lags[i]
		data, mean = armaData(ar, ma, 1.0, timeSteps, startingArray=None)
		np.save(names[i], data)
		np.save(names[i][:-4]+'mean.npy', mean)
		

	sineCoeffs = [10.7, 5.5, -6.25, 7.5, -4.67, 11.2, 2.65, -5.8, 13.7, 9.0]
	sinePeriods = [12.2, 30.3, 4.4, 6.9, 2.6, 9.4, 7.4, 8.3, 2.9, 25.2]

	cosineCoeffs = []#[50.0, 37.0]
	cosinePeriods = []#[1.0, 3.0]

	h10 = gH.generate(sineCoeffs, sinePeriods, cosineCoeffs, cosinePeriods, timeSteps)
	np.save('./single/h10mean.npy', h10)
	errorArray = np.random.normal(0.0, 1.0, timeSteps)
	h10+=errorArray
	np.save('./single/h10.npy', h10)
	

	h3 = gH.generate(sineCoeffs[:3], sinePeriods[:3], cosineCoeffs, cosinePeriods, timeSteps)
	np.save('./single/h3mean.npy', h3)
	errorArray = np.random.normal(0.0, 1.0, timeSteps)
	h3+=errorArray
	np.save('./single/h3.npy', h3)


	sqrt = gT.generate(gT.linearTrendFn, power=.5, displacement=0.0, timeSteps=timeSteps)
	np.save('./single/sqrtmean.npy', sqrt)
	errorArray = np.random.normal(0.0, 1.0, timeSteps)
	sqrt+=errorArray
	np.save('./single/sqrt.npy', sqrt)

	log = gT.generate(gT.logTrendFn, dampening=None, displacement = 0.0, power=None, timeSteps=timeSteps)
	np.save('./single/logmean.npy', log)
	errorArray = np.random.normal(0.0, 1.0, timeSteps)
	log+=errorArray
	np.save('./single/log.npy', log)

def get_key(pair, pref):
	name = ''
	for series in pair:
		index = series.rfind('.')
		name +=(series[9:index])
	return pref+name+'.npy'

def generate_alphas(series):
	num = len(series)
	alphas = []
	total = 0.0

	for i in range(num):
		a = np.random.normal(0.0,1.0)
		total+=a
		alphas.append(a)
	out = []
	for i in alphas:
		out.append(i/total)
	assert abs(sum(out)-1.0)<.00001
	return out


def combine(seriess, pref, dictn):
	key = get_key(seriess, pref)
	series = []
	means = []
	# import pdb; pdb.set_trace()
	for i in seriess:
		series.append(np.load(i))
		means.append(np.load(i[:-4]+'mean.npy'))

	alphas = [0.0]
	while 0.0 in alphas:
		alphas = generate_alphas(series)
	
	assert len(alphas)==len(series)

	out = np.zeros(len(series[0]))
	mean = out.copy()

	for i in range(len(series)):
		out+=alphas[i]*series[i]
		mean+=alphas[i]*means[i]

	meankey=key[:-4]+'mean.npy'

	np.save(key, out)
	np.save(meankey, mean)

	dictn[seriess]={'loc':key, 'alphas':alphas, 'meanloc':meankey, 'series':out, 'meanseries':mean}

def loaded_test(trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF, N, M, nbrSingValuesToKeep, pObservation, means = False):

	key1 = 't1'

	trainProp = 0.9
	M1 = int(trainProp * M)

	mod = SVDModel(key1, nbrSingValuesToKeep, N, M1, probObservation=pObservation, svdMethod='numpy', otherSeriesKeysArray=[], includePastDataOnly=True)
	mod.fit(trainDF)

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

	meanforecastRMSE = tsUtils.rmse(meanTestDF[key1].values, forecastArray)
	obsforecastRMSE = tsUtils.rmse(testDF[key1].values, forecastArray)

	return obsforecastRMSE, meanforecastRMSE

def prophet(testDF, meanTestDF, train2, m):
	m.fit(train2)
	future = m.make_future_dataframe(periods=len(testDF))
	forecast = m.predict(future)
	rmse = tsUtils.rmse(testDF['t1'].values, forecast['yhat'].tolist()[-len(testDF):])
	meanrmse = tsUtils.rmse(meanTestDF['t1'].values, forecast['yhat'].tolist()[-len(testDF):])
	return rmse, meanrmse

def generate(timeSteps, generate_new=True):
	if generate_new: generate_singles(timeSteps)

	series = ['./single/ar25.npy', './single/ar29.npy', './single/ar55.npy', './single/ar59.npy', './single/h10.npy', './single/h3.npy', './single/log.npy', './single/sqrt.npy']
	pairwise = list(combination(series, 2))
	triple = list(combination(series, 3))

	singleSets = []
	for i in series:
		singleSets.append(frozenset([i]))

	pairwiseSets = []
	for i in pairwise:
		pairwiseSets.append(frozenset(i))

	tripleSets = []
	for i in triple:
		tripleSets.append(frozenset(i))

	# pairwiseSets2=[]
	# for i in pairwiseSets:
	# 	elts= []
	# 	for j in i:
	# 		elts.append(j)
	# 	if ('ar' in elts[0] and 'ar' in elts[1]) or ('h' in elts[0] and 'h' in elts[1]) or ('log' in elts[0] and 'sqrt' in elts[1]) or ('sqrt' in elts[0] and 'log' in elts[1]):
	# 		continue
	# 	else:
	# 		pairwiseSets2.append(i)

	tripleSets2=[]
	for i in tripleSets:
		ar = 0
		h = 0
		trend = 0

		for j in i:
			if 'ar' in j: ar+=1
			if 'h' in j: h+=1
			if 'log' in j or 'sqrt' in j: trend+=1

		if ar<2 and h<2 and trend<2:
			tripleSets2.append(i)

	tripleSets = tripleSets2
	# pairwiseSets = pairwiseSets2

	dictn = {}

	for i in singleSets:
		string = tuple(i)[0]
		dictn[i]={'loc': string, 'alphas':[1.0], 'meanloc':string[:-4]+'.npy', 'series':np.load(string), 'meanseries':np.load(string[:-4]+'mean.npy')}

	for i in pairwiseSets:
		combine(i, './pairs/', dictn)

	for i in tripleSets:
		combine(i, './triples/', dictn)

	np.save('dictn.npy', dictn)

	print(dictn)

def test_all(N, M):
	dictn = np.load('dictn.npy').item()
	for key in dictn:
		series = dictn[key]['series']
		meanseries = dictn[key]['meanseries']
		for p in [.4,.6,.8,1.0]:
			trainMasterDF, trainDF, testDF, meanTrainDF, meanTestDF = make_df(series, meanseries, N, M, p, None)
			dictn[key][p]={}
			print 'p='+str(p)
			for singvals in [3]:
				print '     singvals: '+str(singvals)
				obsrmse, meanrmse = loaded_test(trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF, N, M, singvals, 1.0, True)
				print obsrmse
				dictn[key][p][singvals]=(obsrmse, meanrmse)
		# np.save('dictn.npy', dictn)
	np.save('dictn2.npy', dictn)

	for key in dictn:
		loc = dictn[key]['loc']
		meanloc = dictn[key]['mean']
		series = np.load(loc)
		meanseries = np.load(meanloc)
		for p in [.4,.6,.8,1.0]:
			trainMasterDF, trainDF, testDF, meanTrainDF, meanTestDF = make_df(series, meanseries, N, M, p, None)
			train2 = pd.DataFrame(data={'ds':trainDF.index, 'y': trainDF['t1']})
			m=Prophet()
			obsrmse, meanrmse = prophet(testDF, meanTestDF, train2, m)
			dictn[key][p]['prophet']=(obsrmse, meanrmse)

		# np.save('dictn3.npy', dictn)
	np.save('dictn4.npy', dictn)

def test_all2(N, M):
	dictn = np.load('dictn.npy').item()
	for key in dictn:
		series = dictn[key]['series']
		meanseries = dictn[key]['meanseries']
		for p in [.4,.6,.8,1.0]:
			trainMasterDF, trainDF, testDF, meanTrainDF, meanTestDF = make_df(series, meanseries, N, M, p, None)
			dictn[key][p]={}
			print 'p='+str(p)
			for singvals in [3]:
				print '     singvals: '+str(singvals)
				obsrmse, meanrmse = loaded_test(trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF, N, M, singvals, 1.0, True)
				print obsrmse
				dictn[key][p][singvals]=(obsrmse, meanrmse)
			train2 = pd.DataFrame(data={'ds':trainDF.index, 'y': trainDF['t1']})
			m=Prophet()
			obsrmse, meanrmse = prophet(testDF, meanTestDF, train2, m)
			dictn[key][p]['prophet']=(obsrmse, meanrmse)
		# np.save('dictn.npy', dictn)
		np.save('dictn.npy', dictn)
	np.save('dictn2.npy', dictn)

def mult_sing(N,M):
	dictn = np.load('dictn.npy').item()
	for key in dictn:
		series = dictn[key]['series']
		meanseries = dictn[key]['meanseries']
		for p in [.4,.6,.8,1.0]:
			if p not in dictn[key]: dictn[key][p]={}
			if 'list' not in dictn[key][p]: dictn[key][p]['list']=[]
			trainMasterDF, trainDF, testDF, meanTrainDF, meanTestDF = make_df(series, meanseries, N, M, p, None)
			for singvals in [3, 5, 8, 12, 16, 20, 25]:
				print '     singvals: '+str(singvals)
				obsrmse, meanrmse = loaded_test(trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF, N, M, singvals, 1.0, True)
				print obsrmse
				dictn[key][p]['list'].append((obsrmse, meanrmse, singvals))
			train2 = pd.DataFrame(data={'ds':trainDF.index, 'y': trainDF['t1']})
			m=Prophet()
			obsrmse, meanrmse = prophet(testDF, meanTestDF, train2, m)
			dictn[key][p]['prophet']=(obsrmse, meanrmse)
			dictn[key][p]['list']=sorted(dictn[key][p]['list'], key = lambda x:x[1])
		np.save('dictn3.npy', dictn)
	np.save('dictn4.npy', dictn)

# def prophet(dictloc, scale, N, M):
# 	dictn = np.load(dictloc).item()
# 	for key in dictn:
# 		series = dictn[key]['series']
# 		meanseries = dictn[key]['meanseries']
# 		for p in [.4,.6,.8,1.0]:
# 			trainMasterDF, trainDF, testDF, meanTrainDF, meanTestDF = make_df(series, meanseries, N, M, p, None, scale)
# 			print len(trainDF['t1'])
# 			train2 = pd.DataFrame(data={'ds':trainDF.index, 'y': trainDF['t1']})
# 			m=Prophet()
# 			obsrmse, meanrmse = prophet(testDF, meanTestDF, train2, m)
# 			dictn[key][p]['prophet']=(obsrmse, meanrmse)

# 		np.save('dictn30.npy', dictn)
# 	np.save('dictn40.npy', dictn)

def mult_sing_big(N,M):
	dictn = np.load('dictn.npy').item()
	dictn2 = np.load('dictn4.npy').item()
	for key in dictn:
		series = dictn[key]['series']
		meanseries = dictn[key]['meanseries']
		for p in [.4,.6,.8,1.0]:
			if p not in dictn[key]: dictn[key][p]={}
			if 'list' not in dictn[key][p]: dictn[key][p]['list']=[]
			trainMasterDF, trainDF, testDF, meanTrainDF, meanTestDF = make_df(series, meanseries, N, M, p, None, 'minute')
			print('        len df = '+str(len(trainDF['t1'])))

			singvallist = []
			print('        '+str(key))
			print('        '+str(dictn2[key][p]['list'][:3]))
			for score in dictn2[key][p]['list'][:3]:
				singvallist.append(score[2])
				print('        '+str(score[2]))


			for singvals in singvallist:
				print '     singvals: '+str(singvals)
				obsrmse, meanrmse = loaded_test(trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF, N, M, singvals, 1.0, True)
				print obsrmse
				dictn[key][p]['list'].append((obsrmse, meanrmse, singvals))
			train2 = pd.DataFrame(data={'ds':trainDF.index, 'y': trainDF['t1']})
			m=Prophet()
			obsrmse, meanrmse = prophet(testDF, meanTestDF, train2, m)
			dictn[key][p]['prophet']=(obsrmse, meanrmse)
			dictn[key][p]['list']=sorted(dictn[key][p]['list'], key = lambda x:x[1])
		np.save('dictn31.npy', dictn)
	np.save('dictn41.npy', dictn)

def get_imputed(trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF, N, M, nbrSingValuesToKeep, pObservation, means = False):

	key1 = 't1'

	trainProp = 0.9
	M1 = int(trainProp * M)

	mod = SVDModel(key1, nbrSingValuesToKeep, N, M1, probObservation=pObservation, svdMethod='numpy', otherSeriesKeysArray=[], includePastDataOnly=True)
	mod.fit(trainDF)
	return mod.denoisedDF();

def variance(N,M, sv):
	for o, m in [('./single/ar25.npy', './single/ar25mean.npy'), ('./single/h3.npy','./single/h3mean.npy'), ('./single/log.npy', './single/logmean.npy')]:
		print o
		armaObs = np.load(o)
		armaMean = np.load(m)
		trainMasterDF, trainDF, testDF, meanTrainDF, meanTestDF, max, min = make_df_unnorm(armaObs, armaMean, N, M, 1.0, None)
		imputed = get_imputed(trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF, N, M, sv, 1.0, means = False)
		imputedNP = unnormalize(np.ravel(imputed.values), max, min)
		obs = armaObs[:int(.9*N*M)]
		var = np.square(obs-imputedNP)

		obs = plt.plot(obs, 'r')
		mean = plt.plot(imputedNP, 'b')
		
		plt.legend([obs, mean], ['observed', 'mean'])
		plt.show()
		print "mean: "+str(float(np.mean(var)))
		print "stdev: "+str(float(np.std(var)))
		
	  	trainMasterDF, trainDF, testDF, meanTrainDF, meanTestDF = make_df(var, var, N, M, 1.0, None)
	  	
	  	imputed = get_imputed(trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF, N, M,sv, 1.0, means = False)
	  	plt.plot(var, 'r')
	  	plt.show()

def unnormalize(array, max, min):

    diff = 0.5*(min + max)
    div = 0.5 * (max - min)

    array = (array*div + diff)
    return array



	
def generate_singlesSD(timeSteps, SD=1.0):
	lags = [([1.0, .25,.2],[]),
	([1.0, .7875,.1125],[]),
	([-1.0, 1.1, -0.484, 0.107, -0.012, 0.001], []),
	([-1.0, 1.526, -0.931, 0.284, -0.043, 0.003], [])]

	names = ['./single/ar25.npy', './single/ar29.npy', './single/ar55.npy', './single/ar59.npy', './single/h10.npy', './single/h3.npy', './single/log.npy', './single/sqrt.npy']

	for i in range(len(lags)):
		ar, ma = lags[i]
		data, mean = armaData(ar, ma, SD, timeSteps, startingArray=None)
		np.save(names[i], data)
		np.save(names[i][:-4]+'mean.npy', mean)
		

	sineCoeffs = [10.7, 5.5, -6.25, 7.5, -4.67, 11.2, 2.65, -5.8, 13.7, 9.0]
	sinePeriods = [12.2, 30.3, 4.4, 6.9, 2.6, 9.4, 7.4, 8.3, 2.9, 25.2]

	cosineCoeffs = []#[50.0, 37.0]
	cosinePeriods = []#[1.0, 3.0]

	h10 = gH.generate(sineCoeffs, sinePeriods, cosineCoeffs, cosinePeriods, timeSteps)
	np.save('./single/h10mean.npy', h10)
	errorArray = np.random.normal(0.0, SD, timeSteps)
	h10+=errorArray
	np.save('./single/h10.npy', h10)
	

	h3 = gH.generate(sineCoeffs[:3], sinePeriods[:3], cosineCoeffs, cosinePeriods, timeSteps)
	np.save('./single/h3mean.npy', h3)
	errorArray = np.random.normal(0.0, SD, timeSteps)
	h3+=errorArray
	np.save('./single/h3.npy', h3)


	sqrt = gT.generate(gT.linearTrendFn, power=.5, displacement=0.0, timeSteps=timeSteps)
	np.save('./single/sqrtmean.npy', sqrt)
	errorArray = np.random.normal(0.0, SD, timeSteps)
	sqrt+=errorArray
	np.save('./single/sqrt.npy', sqrt)

	log = gT.generate(gT.logTrendFn, dampening=None, displacement = 0.0, power=None, timeSteps=timeSteps)
	np.save('./single/logmean.npy', log)
	errorArray = np.random.normal(0.0, SD, timeSteps)
	log+=errorArray
	np.save('./single/log.npy', log)

if __name__ == '__main__':
	# generate(100*1000)
	# mult_sing(100,1000)ss
	# prophet('dictn4.npy', 'hour', 100, 1000)
	generate_singlesSD(10*1000, 1.0)
	variance(10,1000, 50)
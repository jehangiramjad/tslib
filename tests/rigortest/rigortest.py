import sys, os
from fbprophet import Prophet

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
import tslib.src.data.generateARMAvar as gAv
import tslib.src.data.generateARMAvar as gAvar
from  tslib.src.models.tsSVDModel import SVDModel
from  tslib.src.models.tsALSModel import ALSModel
import tslib.src.tsUtils as tsUtils

from itertools import combinations as combination
from svdpredict import make_df

series = ['./single/ar25.npy', './single/ar29.npy', './single/ar55.npy', './single/ar59.npy', './single/h10.npy', './single/h3.npy', './single/log.npy', './single/sqrt.npy']
pairwise = list(combination(series, 2))
triple = list(combination(series, 3))

pairwiseSets = []
for i in pairwise:
	pairwiseSets.append(frozenset(i))

tripleSets = []
for i in triple:
	tripleSets.append(frozenset(i))

pairwiseSets2=[]
for i in pairwiseSets:
	elts= []
	for j in i:
		elts.append(j)
	if ('ar' in elts[0] and 'ar' in elts[1]) or ('h' in elts[0] and 'h' in elts[1]) or ('log' in elts[0] and 'sqrt' in elts[1]) or ('sqrt' in elts[0] and 'log' in elts[1]):
		continue
	else:
		pairwiseSets2.append(i)

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
pairwiseSets = pairwiseSets2

print len(pairwiseSets)
print len(tripleSets)	

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
	for i in seriess:
		series.append(np.load(i))
	alphas = [0.0]
	while 0.0 in alphas:
		alphas = generate_alphas(series)
	
	assert len(alphas)==len(series)

	out = np.zeros(len(series[0]))

	for i in range(len(series)):
		out+=alphas[i]*series[i]

	np.save(key, out)

	dictn[seriess]={'loc':key, 'alphas':alphas}


def loaded_test(trainMasterDF, trainDF, meanTrainDF, testDF, meanTestDF, N, M, nbrSingValuesToKeep, pObservation, means = True):

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

	forecastRMSE = tsUtils.rmse(testDF[key1].values, forecastArray)

	return forecastRMSE

def prophet(testDF, train2, m):
	m.fit(train2)
	future = m.make_future_dataframe(periods=len(testDF))
	forecast = m.predict(future)
	rmse = tsUtils.rmse(testDF['t1'].values, forecast['yhat'].tolist()[-len(testDF):])
	return rmse


dictn={}
for i in pairwiseSets:
	combine(i, './pairs/', dictn)

for i in tripleSets:
	combine(i, './triples/', dictn)

np.save('dictn.npy', dictn)

for key in dictn:
	loc = dictn[key]['loc']
	series = np.load(loc)
	for p in [.4,.6,.8,1.0]:
		trainMasterDF, trainDF, testDF = make_df(series, series, 100, 1000, p, None)
		dictn[key][p]={}
		print 'p='+str(p)
		for singvals in [3]:
			print '     singvals: '+str(singvals)
			rmse = loaded_test(trainMasterDF, trainDF, trainDF, testDF, testDF, 100, 1000, singvals, 1.0, False)
			print rmse
			dictn[key][p][singvals]=rmse
	np.save('dictn.npy', dictn)
np.save('dictn2.npy', dictn)

for key in dictn:
	loc = dictn[key]['loc']
	series = np.load(loc)
	for p in [.4,.6,.8,1.0]:
		trainMasterDF, trainDF, testDF = make_df(series, series, 100, 1000, p, None)
		train2 = pd.DataFrame(data={'ds':trainDF.index, 'y': trainDF['t1']})
		m=Prophet()
		rmse = prophet(testDF, train2, m)
		dictn[key][p]['prophet']=rmse

	np.save('dictn3.npy', dictn)
np.save('dictn4.npy', dictn)




	








	





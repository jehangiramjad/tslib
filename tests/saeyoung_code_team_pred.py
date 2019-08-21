import sys, os
sys.path.append("../..")
sys.path.append("..")
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import copy
import pickle

from tslib.src import tsUtils
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from tslib.src.synthcontrol.multisyntheticControl import MultiRobustSyntheticControl

import random
from numpy.linalg import svd, matrix_rank, norm
from sklearn import linear_model
import pickle

def hsvt(X, rank): 
	"""
	Input:
		X: matrix of interest
		rank: rank of output matrix
	Output:
		thresholded matrix
	"""
	u, s, v = np.linalg.svd(X, full_matrices=False)
	s[rank:].fill(0)
	return np.dot(u*s, v)

def hsvt_df(X, rank): 
	"""
	Input:
		X: matrix of interest
		rank: rank of output matrix
	Output:
		thresholded matrix
	"""
	u, s, v = np.linalg.svd(X, full_matrices=False)
	s[rank:].fill(0)
	return pd.DataFrame(np.dot(u*s, v), index = X.index, columns = X.columns)

def get_preint_data(X, T0, T, K):
	"""
	Input:
		X: N x KT matrix
		T0: pre-int period
		T: total period
		K: number of metrics
	
	Output:
		X_pre: N x KT0 matrix
	"""
	X_pre = np.array([])
	for k in range(K): 
		if X.ndim > 1:
			X_temp = X[:, k*T:k*T + T0]
		else:
			X_temp = X[k*T:k*T + T0]
		X_pre = np.hstack([X_pre, X_temp]) if X_pre.size else X_temp
	return X_pre

def get_postint_data(X, T0, T, K):
	"""
	Input:
		X: N x KT matrix
		T0: pre-int period
		T: total period
		K: number of metrics
	
	Output:
		X_post: N x K(T-T0) matrix
	"""
	X_post = np.array([])
	for k in range(K): 
		if X.ndim > 1:
			X_temp = X[:, k*T+T0:(k+1)*T]
		else:
			X_temp = X[k*T+T0:(k+1)*T]
		X_post = np.hstack([X_post, X_temp]) if X_post.size else X_temp
	return X_post


def pre_post_split(y, T0, T, num_metrics):
		y_pre = get_preint_data(y, T0, T, num_metrics)
		y_post = get_postint_data(y, T0, T, num_metrics)
		return y_pre, y_post


def approximate_rank(X, t=99):
	"""
	Input:
		X: matrix of interest
		t: an energy threshold. Default (99%)
		
	Output:
		r: approximate rank of Z
	"""
	u, s, v = np.linalg.svd(X, full_matrices=False)
	total_energy = (100*(s**2).cumsum()/(s**2).sum())
	r = list((total_energy>t)).index(True)+1
	return r

def relative_spectrum(X):
	"""
	Input:
		X: matrix of interest
		
	Output:
		list: with % of spectrum explained by first eigenvalues of Z
	"""
	u, s, v = np.linalg.svd(X, full_matrices=False)
	return (s**2)/((s**2).sum())

def donor_prep(X, t):
	"""
	Input:
		X: matrix of interest
		t: threshold
	
	Output:
		thresholded matrix
	"""
	r = approximate_rank(X, thresh)
	print("{} SV = {}% of energy".format(r, t))
	X_hsvt = hsvt(X, r)
	return np.abs(X_hsvt.round())

def mse(y, y_pred):
	return np.sum((y - y_pred) ** 2) / len(y)

def mape(y, y_pred):
	mask = (y != 0)
	return np.mean(np.abs((y - y_pred)[mask] / y[mask]))

def getData(pre1, pre2, metrics, game_ids):
	"""
		pre1 = (string) target or donor
		pre2 = (string) home or away
		metrics = (list) list of metrics
	"""
	prefix = pre1+ "_" + pre2 + "_"
	df = pd.DataFrame()
	for i in range(len(metrics)):
		bucket = pd.read_pickle("../data/nba-hosoi/"+ prefix +metrics[i]+".pkl")
		df = pd.concat([df, bucket], axis = 1)
	df = df[df.index.isin(game_ids)]
	print("DataFrame size ", df.shape, "was created.")
	return df

class mRSC:
	def __init__(self, donor, target, metrics, donor_ids, target_ids, T_0s, singvals): 
		"""
		donor = (df) donor matrix
		target = (df) target_matrix
		metrics = (list) list of metrics in donor/target matrix
		donor_ids = (list) donor ids
		target_ids = (list) target_ids
		T_0s = (list)
		singvals = (int) the number of singular values to keep; 0 if no HSVT
		"""
		if (singvals != 0):
			self.donor = hsvt_df(donor, singvals)
		else:
			self.donor = donor
		self.target = target
		self.metrics = metrics
		self.donor_ids = donor_ids
		self.target_ids = target_ids
		self.num_k = len(self.metrics)
		self.T = int(self.target.shape[1]/self.num_k)
		self.T_0s = T_0s
		self.singvals = singvals
		
		self.pred = [pd.DataFrame(columns=self.target.columns, index=self.target.index)] * len(T_0s)
		self.betas = [pd.DataFrame(columns=self.donor.index, index=self.target.index)] * len(T_0s)
	
	def learn(self, target_id, T_0, method='lr'):
		# treatment unit
		y = self.target[self.target.index == target_id]
		y = y.values.flatten()

		# pre-intervention
		donor_pre = get_preint_data(self.donor.values, T_0, self.T, self.num_k)
		y_pre = get_preint_data(y, T_0, self.T, self.num_k)

		if (method == 'lr'):
			# linear regression
			regr = linear_model.LinearRegression(fit_intercept=False)
			regr.n_jobs = -1
			regr.fit(donor_pre.T, y_pre)
			beta = regr.coef_
			
		elif (method == 'pinv'):
			beta = np.linalg.pinv(donor_pre.T).dot(y_pre)
			
		else:
			raise ValueError("Invalid method.")
		
		i = np.where(np.array(self.T_0s) == T_0)[0][0]
		
		# beta
		updated_beta = self.betas[i].copy()
		updated_beta[updated_beta.index == target_id] = [beta]
		self.betas[i] = updated_beta
		
		# prediction
		prediction = self.donor.T.dot(beta).values
		updated_pred = self.pred[i].copy()
		updated_pred[updated_pred.index == target_id] = [prediction]
		self.pred[i]= updated_pred

def run():
	""" Separte, with 5 Metrics, w/o HSVT """

	""" construct the matrix """
	"""
		Donor: 2013 - 2016 season
		Target: 2017 season
	"""

	# experiment prarams
	train_pcts = [0.1, 0.25, 0.5, 0.75, 0.9]
	freq = 15
	T = int(12*60*4/freq + 1)
	T_0s = [int(np.ceil(train_pct * T)) for train_pct in train_pcts]
	singvals= 0
	donor_ids = np.array(pd.read_pickle('../data/nba-hosoi/donor_ids.pkl'))
	target_ids = np.array(pd.read_pickle('../data/nba-hosoi/target_ids.pkl'))
	metrics = ['points','assists', 'rebounds', 'bs', 'fouls']

	# import data
	donor_home = getData("donor", "home", metrics, donor_ids)
	donor_away = getData("donor", "away", metrics, donor_ids)

	target_home = getData("target", "home", metrics, target_ids)
	target_away = getData("target", "away", metrics, target_ids)

	# construct model
	mRSC_home = mRSC(donor_home, target_home, metrics, donor_ids, target_ids, T_0s, singvals)
	mRSC_away = mRSC(donor_away, target_away, metrics, donor_ids, target_ids, T_0s, singvals)

	# run
	for i in range(len(train_pcts)):
		T_0 = T_0s[i]
		print("start: ", str(i+1), " / 5")
		# for each game in target
		for target_id in target_ids:
			mRSC_home.learn(target_id, T_0, method='lr')
			mRSC_away.learn(target_id, T_0, method='lr')

	filename = "result_team/saeyoung_no_hsvt_home.pkl"
	with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
		pickle.dump(mRSC_home.pred, f)
	filename = "result_team/saeyoung_no_hsvt_away.pkl"
	with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
		pickle.dump(mRSC_away.pred, f)
		


# Run it only if called from the command line
if __name__ == '__main__':
	run()
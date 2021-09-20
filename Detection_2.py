#!/usr/bin/python3

import numpy as np
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from math import *
# from LSTM import LSTM_classification
import pandas as pd

epsilon = 1e-3
index = 0
dataset = ['full','ham25','ham50','spam25','spam50']
CleanHam = 8651
CleanSpam = 8835
folder = 1

def origin_Prob(XTrain, Para):
	# E step: P(j|xi) 
	# origin: NTrain X M P(j|xi)
	# multiplication may introduce underflow, resulting origin_prob = 0.
	# So we take log on Prob, compute log-likelihood of each xi, make them positive and restore to prob finally
	(M, alpha, comp_spec_Prob, shared_Prob, V) = Para
	NTrain, D = XTrain.shape

	Prob = np.add(np.multiply(comp_spec_Prob, V), np.multiply(shared_Prob, np.logical_not(V)))  # M X D: multinomial para used by each component
	Prob[Prob == 0] = epsilon
	Prob = np.log(Prob)

	tmp = XTrain.dot(Prob.T)   # N X M
	tmp = tmp - np.max(tmp, axis = 1).reshape([NTrain,1])  
	tmp = np.exp(tmp)  # restore to prob
	# tmp[tmp < epsilon] = 0
	tmp = np.multiply(tmp, alpha.reshape([1,M]))
	origin = np.divide(tmp, tmp.sum(axis = 1).reshape([NTrain,1]))

	return origin

def log_factorial(mat):

	# Approximation for logn! given by Srinivasa Ramanujan
	# mat[mat==0] = 1
	# lf = np.log(mat)
	# lf = np.multiply(mat, lf)
	# lf -= mat
	# lf += log(pi)/2
	# lf += np.log(np.multiply(np.multiply(1+2*mat, 4*mat) + 1, mat))/6
	# Stirling's approximation log(n!)=nlogn-n
	mat.data = mat.data*np.log(mat.data)-mat.data
	mat[mat==-1]=0

	return mat

def comp_log_likelihood(features, Para):

	# compute log p[x|j]
	(M, alpha, comp_spec_Prob, shared_Prob, V) = Para
	D = comp_spec_Prob.shape[1]
	Prob= np.add(np.multiply(comp_spec_Prob, V), np.multiply(shared_Prob, np.logical_not(V)))
	# Prob = comp_spec_Prob
	# print(Prob.shape)
	Prob[Prob==0] = epsilon
	Prob = np.log(Prob)  # M X D
	
	N = features.shape[0]
	factorial = log_factorial(features.copy()).sum(axis = 1)   # N X 1
	factorial = log_factorial(features.sum(axis=1)) - factorial.reshape([N,1])
	log_likelihood = features.dot(Prob.T).reshape([N,M])   # N X M
	log_likelihood = np.add(factorial.reshape([N,1]), log_likelihood)  # N X M

	return log_likelihood

def mixture_log_likelihood(features, Para):

	# compute p[xi|c]=E[log(p[xi|c])]+entropy
	N,D = features.shape
	M = Para[0]
	alpha = Para[1]
	alpha[alpha==0] = epsilon
	log_alpha = np.log(alpha)
	P = origin_Prob(features, Para)
	P[P==0] = epsilon

	# E[log(p[xi|c])]+entropy
	compLikelihood = comp_log_likelihood(features, Para)    # N X M: log(p[xi|j,c])
	compLikelihood = np.add(log_alpha.reshape([1,M]), compLikelihood)    # N X M: log(aj)+log(p[xi|j,c])
	compLikelihood = np.multiply(P, compLikelihood).sum(axis = 1)   # N X 1: 
	compLikelihood = compLikelihood.reshape([N,1]) - np.multiply(P, np.log(P)).sum(axis = 1).reshape([N,1])  # expected loglikelihood + entropy for all xi

	return compLikelihood

def prune_component(Para, index):

	if isinstance(index, int):
		s = 1
	else:
		s = index.size
	(M, alpha, comp_spec_Prob, shared_Prob, V) = Para
	M_new = M - s
	alpha_new = np.delete(alpha, index)
	alpha_new = alpha_new/alpha_new.sum()
	comp_spec_Prob_new = np.delete(comp_spec_Prob, index, axis = 0)
	V_new = np.delete(V, index, axis = 0)
	Para_new = [M_new, alpha_new, comp_spec_Prob_new, shared_Prob, V_new]

	return Para_new

def classifiers(Train, labelsTrain, Test, labelsTest, classifier):

	if classifier == 'NB':
		clf = MultinomialNB()
		clf.fit(Train, labelsTrain)
		pred = clf.predict(Test)
		acc = metrics.accuracy_score(labelsTest, pred)
		print(acc)
		confusion = metrics.confusion_matrix(labelsTest, pred)
		print(confusion)
	elif classifier == 'SVM':
		# clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
		clf = svm.LinearSVC()
		clf.fit(Train, labelsTrain)
		pred = clf.predict(Test)
		acc = metrics.accuracy_score(labelsTest, pred)
		print(acc)
		confusion = metrics.confusion_matrix(labelsTest, pred)
		print(confusion)
		decision_function = clf.decision_function(Train)
		support_vector_indices = np.where((2 * labelsTrain - 1) * decision_function <= 1)[0]
		# return support_vector_indices
	elif classifier == 'LR':
		clf = linear_model.LogisticRegression(random_state=0, max_iter=2000, tol=1e-3)
		# clf = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
		clf.fit(Train, labelsTrain)
		pred = clf.predict(Test)
		acc = metrics.accuracy_score(labelsTest, pred)
		print(acc)
		confusion = metrics.confusion_matrix(labelsTest, pred)
		print(confusion)

	return acc

def update_switch(features, N, Para, j):

	# v_j = Para[4][j].copy()
	D = features.shape[1]
	Para_j = [1, 1, Para[2][j].reshape((1,D)), Para[3], Para[4][j].reshape((1,D))]
	BIC_old = model_cost(Para, N) - comp_log_likelihood(features, Para_j).sum()

	for k in range(D):
		# trail-flip v_jk
		Para[4][j,k] = np.logical_not(Para[4][j,k])
		Para_j = [1, 1, Para[2][j].reshape((1,D)), Para[3], Para[4][j].reshape((1,D))]
		BIC_new = model_cost(Para, N) - comp_log_likelihood(features, Para_j).sum()
		if BIC_new < BIC_old:
			BIC_old = BIC_new
		else:
			Para[4][j,k] = np.logical_not(Para[4][j,k])
	print(Para[4][j].sum())

	return Para

def model_cost(Para, N):

	v_k = Para[4].sum(axis=0)
	M = Para[0]
	cost = 0
	code_len = 1/2*log(N)

	cost += v_k.sum() * code_len  # cost of component specific para
	cost += np.where(v_k<M)[0].size * code_len  # cost of component shared para
	cost += np.where(np.logical_and(v_k>0,v_k<M))[0].size * M*log(2)  # cost of switches
	cost += (M-1) * code_len  # cost of component weights

	return cost

def adv_data(j, Para_Spam, Para_Ham, Train, ind, s):

	Para_Spam_exclude_j = prune_component(Para_Spam, j)
	# likelihood_under_ham = np.array(mixture_log_likelihood(Train[ind,:], Para_Ham)).reshape(ind.size,)
	# likelihood_under_spam_exclue_j = np.array(mixture_log_likelihood(Train[ind,:], Para_Spam_exclude_j)).reshape(ind.size,) # component fits x most excluding j
	# w_s = likelihood_under_ham > likelihood_under_spam_exclue_j
	# w_h = likelihood_under_spam_exclue_j > likelihood_under_ham
	likelihood_under_ham = comp_log_likelihood(Train[ind,:], Para_Ham).max(axis=1)
	likelihood_under_spam_exclue_j = comp_log_likelihood(Train[ind,:], Para_Spam_exclude_j).max(axis=1)
	w_s = np.array(likelihood_under_ham > likelihood_under_spam_exclue_j).reshape(ind.size,)
	w_h = np.array(likelihood_under_spam_exclue_j > likelihood_under_ham).reshape(ind.size,)
	tmp = np.logical_or(np.logical_and(s[ind],np.logical_not(w_s)) , np.logical_and(np.logical_not(s[ind]),w_h))

	return w_s, w_h, tmp

def complete_BIC_for_2_cases(j, Para_Spam, Para_Ham, Train, ind, N_spam, s, codelen = 1):
	
	spam_in_j = Train[ind,:]
	N, D = Train.shape
	N_ham = N-N_spam
	Para_j = [1, 1, Para_Spam[2][j, :].reshape((1,D)),Para_Spam[3], Para_Spam[4][j, :].reshape((1,D))]
	# likelihood_under_ham = np.array(mixture_log_likelihood(spam_in_j, Para_Ham)).reshape(ind.size,)
	likelihood_under_ham = comp_log_likelihood(spam_in_j, Para_Ham).max(axis=1)

	# case 1, component j is clean
	# likelihood_under_spam = np.array(mixture_log_likelihood(spam_in_j, Para_Spam)).reshape(ind.size,)
	likelihood_under_spam = comp_log_likelihood(spam_in_j, Para_j)
	LC = likelihood_under_spam.sum()  # (N_j^s,1)
	BIC_1 = -LC
	# print('case 1: {}'.format(BIC_1))

	# case 3, component j is poisoned, revise j
	Para_Spam_exclude_j = prune_component(Para_Spam, j)
	likelihood_under_spam_exclue_j = comp_log_likelihood(spam_in_j, Para_Spam_exclude_j).max(axis=1)
	w_s = np.array(likelihood_under_ham > likelihood_under_spam_exclue_j).reshape(ind.size,)
	w_h = np.array(likelihood_under_spam_exclue_j > likelihood_under_ham).reshape(ind.size,)
	tmp = np.logical_or(np.logical_and(s[ind],np.logical_not(w_s)) , np.logical_and(np.logical_not(s[ind]),w_h))
	# tmp = T => spam, tmp = F => ham
	# tmp2 = np.logical_and(s[ind],tmp)
	surviving_n = tmp.sum()
	# print(j, surviving_n)
	if surviving_n > 0 and surviving_n < tmp.size:
		# Para_Spam_update_j = [Para_Spam[0], Para_Spam[1].copy(), Para_Spam[2].copy(), Para_Spam[3].copy(), Para_Spam[4].copy()]
		# Para_Spam_update_j[2][j] = spam_in_j[tmp, :].sum(axis=0) / spam_in_j[tmp, :].sum() # revise j
		Para_j_update = spam_in_j[tmp, :].sum(axis=0) / spam_in_j[tmp, :].sum()
		# Para_j_update = spam_in_j[tmp, :].sum(axis=0) / spam_in_j[tmp, :].sum()
		Para_j_update = [1, 1, Para_j_update, Para_Spam[3], Para_Spam[4][j, :].reshape((1,D))]
		# change_of_complexity_2 = Para_Spam[4][j].sum()*(-1/2*log(ind.size) + 1/2*log(surviving_n))
		if codelen == 1:
			change_of_complexity_2 = Para_Spam[4][j].sum()*(-1/2*log(ind.size) + 1/2*log(surviving_n))
		else:
			change_of_complexity_2 = model_cost(Para_Spam, N_spam-np.logical_not(tmp).sum()) - model_cost(Para_Spam, N_spam)
			change_of_complexity_2 += model_cost(Para_Ham, N_ham+np.logical_not(tmp).sum()) - model_cost(Para_Ham, N_ham)
		# likelihood_under_spam_2 = np.array(mixture_log_likelihood(spam_in_j, Para_Spam_update_j)).reshape(ind.size,)
		likelihood_under_spam_2 = comp_log_likelihood(spam_in_j, Para_j_update)
		LC = likelihood_under_spam_2[tmp].sum() + likelihood_under_ham[np.logical_not(tmp)].sum()
		BIC_3 = change_of_complexity_2 - LC
		# BIC_3 =  - LC
		# print('case 3: {}, remain {}, remove {}, cost {}, LC {}'.format(BIC_3-BIC_1, tmp.sum(), np.logical_not(tmp).sum(), change_of_complexity_2, -LC))
	else:
		BIC_3 = float('inf')
		# print('case 3: {}, remain {}, remove {}'.format(BIC_3-BIC_1, tmp.sum(), np.logical_not(tmp).sum()))

	if BIC_1 < BIC_3:
		return 0, w_s, w_h, tmp, 'unpoisoned'
	else:
		return BIC_3-BIC_1, w_s, w_h, tmp, 'revise'


def complete_BIC_for_3_cases(j, Para_Spam, Para_Ham, Train, ind, N_spam, s, codelen = 1):
	
	spam_in_j = Train[ind,:]
	N, D = Train.shape
	N_ham = N-N_spam
	Para_j = [1, 1, Para_Spam[2][j, :].reshape((1,D)),Para_Spam[3], Para_Spam[4][j, :].reshape((1,D))]
	likelihood_under_ham = comp_log_likelihood(spam_in_j, Para_Ham).max(axis=1)

	# case 1, component j is clean
	likelihood_under_spam = comp_log_likelihood(spam_in_j, Para_j)
	LC = likelihood_under_spam.sum()  # (N_j^s,1)
	BIC_1 = -LC
	# print('case 1: {}'.format(BIC_1))

	# case 2, component j is poisoned, remove j
	Para_Spam_exclude_j = prune_component(Para_Spam, j)
	likelihood_under_spam_exclue_j = comp_log_likelihood(spam_in_j, Para_Spam_exclude_j).max(axis=1)
	w_s = np.array(likelihood_under_ham > likelihood_under_spam_exclue_j).reshape(ind.size,)
	w_h = np.array(likelihood_under_spam_exclue_j > likelihood_under_ham).reshape(ind.size,)
	
	tmp = np.logical_or(np.logical_and(s[ind],np.logical_not(w_s)) , np.logical_and(np.logical_not(s[ind]),w_h))
	# print(tmp.size, ind.size)
	# tmp = T => spam, tmp = F => ham
	if np.logical_not(tmp).sum() > 0:
		if codelen == 1:
			# code length 1/2logNj 
			# change_of_complexity = -Para_Spam[4][j].sum()*1/2*log(ind.size) -D*log(2)  # cost of theta_j and v_j
			change_of_complexity = -Para_Spam[4][j].sum()*1/2*log(ind.size)  # cost of theta_j
		else:
			# code length 1/2logN
			change_of_complexity = model_cost(Para_Spam_exclude_j, N_spam-np.logical_not(tmp).sum()) - model_cost(Para_Spam, N_spam)
			change_of_complexity += model_cost(Para_Ham, N_ham+np.logical_not(tmp).sum()) - model_cost(Para_Ham, N_ham)
		LC = likelihood_under_spam_exclue_j[tmp].sum() + likelihood_under_ham[np.logical_not(tmp)].sum()
		BIC_2 = change_of_complexity - LC
		# print(likelihood_under_spam[tmp].sum(), likelihood_under_spam_exclue_j[tmp].sum())
		# print('case 2: {}, remain {}, remove {}, cost {}, LC {}'.format(BIC_2-BIC_1, tmp.sum(), np.logical_not(tmp).sum(), change_of_complexity, -LC))
	else:
		BIC_2 = float('inf')
	# print(change_of_complexity, LC)
	# print(ind.size, tmp.sum(), ind.size-tmp.sum())
	# print(BIC_1, -LC)
	# print(j, likelihood_under_spam[tmp].sum(), likelihood_under_spam_exclue_j[tmp].sum())
	# print(likelihood_under_spam[np.logical_not(tmp)].sum(), likelihood_under_ham[np.logical_not(tmp)].sum())
	# BIC_2 = -LC

	# case 3, component j is poisoned, revise j
	tmp2 = np.logical_and(s[ind],tmp)
	# surviving_n = tmp.sum()
	surviving_n = tmp2.sum()
	# print(j, surviving_n)
	if np.logical_not(tmp).sum() > 0 and surviving_n > 0:
	# if tmp2.sum()< 0:
		# Para_Spam_update_j = [Para_Spam[0], Para_Spam[1].copy(), Para_Spam[2].copy(), Para_Spam[3].copy(), Para_Spam[4].copy()]
		# Para_Spam_update_j[2][j] = spam_in_j[tmp, :].sum(axis=0) / spam_in_j[tmp, :].sum() # revise j
		# Para_Spam_update_j = update_switch(spam_in_j[tmp], N_spam-np.logical_not(tmp).sum(), Para_Spam_update_j, j)
		# Para_j_update = spam_in_j[tmp, :].sum(axis=0) / spam_in_j[tmp, :].sum()
		Para_j_update = spam_in_j[tmp2, :].sum(axis=0) / spam_in_j[tmp2, :].sum()
		# shared_update = 
		Para_j_update = [1, 1, Para_j_update, Para_Spam[3], Para_Spam[4][j, :].reshape((1,D))]
		if codelen == 1:
			change_of_complexity_2 = Para_Spam[4][j].sum()*(-1/2*log(ind.size) + 1/2*log(surviving_n))
		else:
			change_of_complexity_2 = model_cost(Para_Spam, N_spam-np.logical_not(tmp).sum()) - model_cost(Para_Spam, N_spam)
			change_of_complexity_2 += model_cost(Para_Ham, N_ham+np.logical_not(tmp).sum()) - model_cost(Para_Ham, N_ham)
		# likelihood_under_spam_2 = np.array(mixture_log_likelihood(spam_in_j, Para_Spam_update_j)).reshape(ind.size,)
		likelihood_under_spam_2 = comp_log_likelihood(spam_in_j, Para_j_update)
		LC = likelihood_under_spam_2[tmp].sum() + likelihood_under_ham[np.logical_not(tmp)].sum()
		BIC_3 = change_of_complexity_2 - LC
		# print(j, likelihood_under_spam_2[tmp].sum(), likelihood_under_spam_exclue_j[tmp].sum())
		# print(likelihood_under_spam_2[tmp].sum())
		# BIC_3 =  - LC
		# print('case 3: {}, remain {}, remove {}, cost {}, LC {}'.format(BIC_3-BIC_1, tmp.sum(), np.logical_not(tmp).sum(), change_of_complexity_2, -LC))
	else:
		BIC_3 = float('inf')
		# print('case 3: {}, remain {}, remove {}'.format(BIC_3-BIC_1, tmp.sum(), np.logical_not(tmp).sum()))

	# if BIC_2 < BIC_3:
	# 	return BIC_2-BIC_1, w_s, w_h, tmp, 'remove'
	# else:
	# 	return BIC_3-BIC_1, w_s, w_h, tmp, 'revise'
	
	if BIC_1 == min(BIC_1, BIC_2, BIC_3):
		return 0, w_s, w_h, tmp, 'unpoisoned'
	else:
		if BIC_2 == min(BIC_1, BIC_2, BIC_3):
			return BIC_2-BIC_1, w_s, w_h, tmp, 'remove'
		else:
			return BIC_3-BIC_1, w_s, w_h, tmp, 'revise'

def sanitization(Para_Spam, Para_Ham, Train, X_s, X_h, w_s, w_h, s, phase, f, Attacks):
	
	counter = 0
	counter_remove_ham = 0
	counter_remove_spam = 0
	counter_revise_ham = 0
	counter_revise_spam = 0

	while True:
				
		counter += 1
		print('iteration {}'.format(counter))
		# if counter == 51:
		# 	break
		
		# spam
		BIC_max_change_spam = np.ones(f)*(-1000)
		component_min_spam = np.ones(f, dtype=int)*(-1)
		para_min_spam = [[]]*f
		N_spam = sum([samples.size for samples in X_s]) # number of samples in spam
		BIC_change_spam = np.zeros(Para_Spam[0])
		
		for j in range(Para_Spam[0]):

			ind = X_s[j]  # global index of samples in component j
			# spam_in_j = Train[ind,:] # samples in component j
			if phase == 1:
				delta_BIC, _, _, _, order = complete_BIC_for_2_cases(j, Para_Spam, Para_Ham, Train, ind, N_spam, s, 2)
			else:
				delta_BIC, _, _, _, order = complete_BIC_for_3_cases(j, Para_Spam, Para_Ham, Train, ind, N_spam, s, 2)
			# if Para_Spam[1][j] < 1:
			# 	delta_BIC, _, _, _, order = complete_BIC_for_3_cases(j, Para_Spam, Para_Ham, Train, ind, N_spam, s, 2)
			# else:
			# 	delta_BIC, _, _, _, order = complete_BIC_for_2_cases(j, Para_Spam, Para_Ham, Train, ind, N_spam, s, 2)
			# print(j,BICs[1]-BICs[0],BICs[2]-BICs[0])
			BIC_change_spam[j] = delta_BIC

			flag = BIC_max_change_spam.argmax()
			if delta_BIC < BIC_max_change_spam[flag]: 
				print('update spam {}, {}'.format(j,order))
				BIC_max_change_spam[flag] = delta_BIC
				component_min_spam[flag] = j
				# para_min_spam[flag] = [order, tmp.copy(), w_s_new.copy(), w_h_new.copy()]
				para_min_spam[flag] = [order]
		# sort component index in descending
		comp_ind = np.argsort(-component_min_spam)
		component_min_spam = component_min_spam[comp_ind]
		BIC_max_change_spam = BIC_max_change_spam[comp_ind]
		para_min_spam = np.array(para_min_spam, dtype=object)[comp_ind]
		print(component_min_spam)
		
		# ham
		BIC_max_change_ham = np.ones(f)*(-1000)
		component_min_ham = np.ones(f, dtype=int)*(-1)
		para_min_ham = [[]]*f
		N_ham = sum([samples.size for samples in X_h]) # number of samples in ham
		BIC_change_ham = np.zeros(Para_Ham[0])
		
		for j in range(Para_Ham[0]):

			ind = X_h[j]  # global index of samples in component j
			# ham_in_j = Train[ind,:] # samples in component j
			if phase == 1:
				delta_BIC, _, _, _, order = complete_BIC_for_2_cases(j, Para_Ham, Para_Spam, Train, ind, N_ham, np.logical_not(s), 2)
			else:
				delta_BIC, _, _, _, order = complete_BIC_for_3_cases(j, Para_Ham, Para_Spam, Train, ind, N_ham, np.logical_not(s), 2)
			# if Para_Ham[1][j] < 1:
			# 	delta_BIC, _, _, _, order = complete_BIC_for_3_cases(j, Para_Ham, Para_Spam, Train, ind, N_ham, np.logical_not(s), 2)
			# else:
			# 	delta_BIC, _, _, _, order = complete_BIC_for_2_cases(j, Para_Ham, Para_Spam, Train, ind, N_ham, np.logical_not(s), 2)
			# print(j,BICs[1]-BICs[0],BICs[2]-BICs[0])
			BIC_change_ham[j] = delta_BIC

			flag = BIC_max_change_ham.argmax()
			if delta_BIC < BIC_max_change_ham[flag]: 
				print('update ham {}, {}'.format(j,order))
				BIC_max_change_ham[flag] = delta_BIC
				component_min_ham[flag] = j
				# para_min_ham[flag] = [order, tmp.copy(), w_s_new.copy(), w_h_new.copy()]
				para_min_ham[flag] = [order]
		# sort component index in descending
		comp_ind = np.argsort(-component_min_ham)
		component_min_ham = component_min_ham[comp_ind]
		BIC_max_change_ham = BIC_max_change_ham[comp_ind]
		para_min_ham = np.array(para_min_ham, dtype=object)[comp_ind]
		print(component_min_ham)
		
		####### decide which class to be sanitized
		# print(BIC_max_change_ham, BIC_max_change_spam)
		# if BIC_max_change_ham.sum() < BIC_max_change_spam.sum():
		# 	# print(BIC_max_change_ham, BIC_2_ham)
		# 	component_min_spam = np.ones(f)*(-1)
		# else:
		# 	# print(BIC_max_change_spam, BIC_2_spam)
		# 	component_min_ham = np.ones(f)*(-1)
		##### f = 1
		# if BIC_change_spam.sum() != 0:
		# 	avg_BIC_change_spam = BIC_change_spam.sum()/(np.where(BIC_change_spam!=0)[0].size)
		# else:
		# 	avg_BIC_change_spam = 0
		# if BIC_change_ham.sum() != 0:
		# 	avg_BIC_change_ham = BIC_change_ham.sum()/(np.where(BIC_change_ham!=0)[0].size)
		# else:
		# 	avg_BIC_change_ham = 0
		# avg_BIC_change_spam = BIC_change_spam.sum()/BIC_change_spam.size
		# avg_BIC_change_ham = BIC_change_ham.sum()/BIC_change_ham.size
		# print(avg_BIC_change_spam, avg_BIC_change_ham)
		# if avg_BIC_change_spam < avg_BIC_change_ham:
		# print(BIC_change_spam.sum(), BIC_change_ham.sum())
		# if BIC_change_spam.sum() < BIC_change_ham.sum():
		# print(BIC_max_change_spam.sum(), BIC_max_change_ham.sum())
		if BIC_max_change_spam.sum() < BIC_max_change_ham.sum():
			component_min_ham = np.ones(f)*(-1)
		else:
			component_min_spam = np.ones(f)*(-1)

		for flag in range(f):
			if component_min_spam[flag] != -1: 

				# print(BIC_max_change_spam[flag])
				# print(BIC_max_change_spam, component_min_spam)
				ind = X_s[component_min_spam[flag]]
				w_s_new, w_h_new, tmp = adv_data(component_min_spam[flag], Para_Spam, Para_Ham, Train, ind, s)
				w_s[ind] = w_s_new
				w_h[ind] = w_h_new
				tmp2 = np.logical_and(tmp, s[ind])
				# print(tmp.size)
				print('total attack: {}, total samples:{}'.format(Attacks[ind].sum(),ind.size))
				attack = np.logical_or(np.logical_and(s[ind], w_s[ind]), np.logical_and(np.logical_not(s[ind]), w_h[ind]))
				print('True :{}, False : {}'.format((attack*Attacks[ind]).sum(), (attack*np.logical_not(Attacks[ind])).sum()))
				# print(para_min_spam[1].sum(), ind.size-para_min_spam[1].sum())

				# re-distribute samples with tmp == False to ham components
				# HamClusters = np.array(origin_Prob(Train[ind,:], Para_Ham).argmax(axis=1)).reshape(ind.size,)
				print('redistribute {} samples to ham'.format(np.logical_not(tmp).sum()))
				HamClusters = np.array(comp_log_likelihood(Train[ind,:], Para_Ham).argmax(axis=1)).reshape(ind.size,)
				for k in range(Para_Ham[0]):
					ind_k_local = np.where(np.logical_and(HamClusters == k, tmp == False))[0]
					if ind_k_local.size > 0:
						X_h[k] = np.concatenate((X_h[k],ind[ind_k_local]))

				if para_min_spam[flag][0] == 'remove':
					# remove j and redistribute samples with tmp == True to other spam components
					counter_remove_spam += 1
					print('remove spam component {} weight {}'.format(component_min_spam[flag], Para_Spam[1][component_min_spam[flag]]))
					Para_Spam = list(prune_component(Para_Spam, component_min_spam[flag]))
					del X_s[component_min_spam[flag]]
					# SpamClusters = np.array(origin_Prob(Train[ind,:], Para_Spam).argmax(axis=1)).reshape(ind.size,)
					SpamClusters = np.array(comp_log_likelihood(Train[ind,:], Para_Spam).argmax(axis=1)).reshape(ind.size,)
					for k in range(Para_Spam[0]):
						ind_k_local = np.where(np.logical_and(SpamClusters == k, tmp == True))[0]
						if ind_k_local.size > 0:
							X_s[k] = np.concatenate((X_s[k],ind[ind_k_local]))

				elif para_min_spam[flag][0] == 'revise': 
					# upate para_j and Xs[j]
					counter_revise_spam += 1
					print('revise spam component {} weight {}'.format(component_min_spam[flag], Para_Spam[1][component_min_spam[flag]]))
					# Para_Spam[2][component_min_spam,:] = para_min_spam[1].copy()
					# Para_Spam[2][component_min_spam[flag],:] = Train[ind][tmp].sum(axis=0) / Train[ind][tmp].sum()
					Para_Spam[2][component_min_spam[flag],:] = Train[ind][tmp2].sum(axis=0) / Train[ind][tmp2].sum()
					# Para_Spam = update_switch(Train[ind][tmp], N_spam-np.logical_not(tmp).sum(), Para_Spam, component_min_spam[flag])
					# print(X_s[component_min_spam].size)
					X_s[component_min_spam[flag]] = X_s[component_min_spam[flag]][tmp]
					# print(X_s[component_min_spam].size)
				else:
					print('wrong')
					break

				Para_Spam[1] = np.array([samples.size for samples in X_s])/sum([samples.size for samples in X_s])

		for flag in range(f):
			if component_min_ham[flag] != -1:

				# print(BIC_max_change_ham[flag])
				# print(BIC_max_change_ham, component_min_ham)
				ind = X_h[component_min_ham[flag]]
				w_h_new, w_s_new, tmp = adv_data(component_min_ham[flag], Para_Ham, Para_Spam, Train, ind, np.logical_not(s))
				w_s[ind] = w_s_new
				w_h[ind] = w_h_new
				tmp2 = np.logical_and(tmp, np.logical_not(s[ind]))
				# print(tmp2.sum(), tmp.size)
				print('total attack: {}, total samples:{}'.format(Attacks[ind].sum(), ind.size))
				attack = np.logical_or(np.logical_and(s[ind], w_s[ind]), np.logical_and(np.logical_not(s[ind]), w_h[ind]))
				print('True :{}, False : {}'.format((attack*Attacks[ind]).sum(), (attack*np.logical_not(Attacks[ind])).sum()))
				# print(para_min_ham[1].sum(), ind.size-para_min_ham[1].sum())

				# re-distribute samples with tmp == False to spam components
				print('redistribute {} samples to spam'.format(np.logical_not(tmp).sum()))
				# SpamClusters = np.array(origin_Prob(Train[ind,:], Para_Spam).argmax(axis=1)).reshape(ind.size,)
				SpamClusters = np.array(comp_log_likelihood(Train[ind,:], Para_Spam).argmax(axis=1)).reshape(ind.size,)
				for k in range(Para_Spam[0]):
					ind_k_local = np.where(np.logical_and(SpamClusters == k, tmp == False))[0]
					if ind_k_local.size > 0:
						X_s[k] = np.concatenate((X_s[k],ind[ind_k_local]))

				if para_min_ham[flag][0] == 'remove':
					counter_remove_ham += 1
					# remove j and redistribute samples with tmp == True to other spam components
					print('remove ham component {} weight {}'.format(component_min_ham[flag], Para_Ham[1][component_min_ham[flag]]))
					Para_Ham = list(prune_component(Para_Ham, component_min_ham[flag]))
					del X_h[component_min_ham[flag]]
					# HamClusters = np.array(origin_Prob(Train[ind,:], Para_Ham).argmax(axis=1)).reshape(ind.size,)
					HamClusters = np.array(comp_log_likelihood(Train[ind,:], Para_Ham).argmax(axis=1)).reshape(ind.size,)
					for k in range(Para_Ham[0]):
						ind_k_local = np.where(np.logical_and(HamClusters == k, tmp == True))[0]
						if ind_k_local.size > 0:
							X_h[k] = np.concatenate((X_h[k],ind[ind_k_local]))
				elif para_min_ham[flag][0] == 'revise': 
					# upate para_j and Xs[j]
					counter_revise_ham += 1
					print('revise ham component {} weight {}'.format(component_min_ham[flag], Para_Ham[1][component_min_ham[flag]]))
					# Para_Ham[2][component_min_ham,:] = para_min_ham[1].copy()
					# Para_Ham[2][component_min_ham[flag],:] = Train[ind][tmp].sum(axis=0) / Train[ind][tmp].sum()
					Para_Ham[2][component_min_ham[flag],:] = Train[ind][tmp2].sum(axis=0) / Train[ind][tmp2].sum()
					# Para_Ham = update_switch(Train[ind][tmp], N_ham-np.logical_not(tmp).sum(), Para_Ham, component_min_ham[flag])
					X_h[component_min_ham[flag]] = X_h[component_min_ham[flag]][tmp]
				else:
					print('wrong')
					break

				Para_Ham[1] = np.array([samples.size for samples in X_h])/sum([samples.size for samples in X_h])
	
		# check
		# print(sum([samples.size for samples in X_h] + [samples.size for samples in X_s]))
		# converge?
		if component_min_spam.sum() == -1*f and component_min_ham.sum() == -1*f:
			break

	print('remove spam: {}, revise spam: {}, remove ham: {}, revise ham: {}'.format(counter_remove_spam, counter_revise_spam, counter_remove_ham, counter_revise_ham))

	return Para_Spam, Para_Ham, X_s, X_h, w_s, w_h

def detection_one_at_a_time(Nham, Nspam, Train, Para_Ham, Para_Spam, Attacks):

	# 1. initialization
	print("initialization")
	(N, D) = Train.shape
	w_s = np.zeros(N, dtype = bool) # 1=>attack
	w_h = np.zeros(N, dtype = bool) # 1=>attack
	s = np.concatenate((np.zeros(Nham, dtype = bool), np.ones(Nspam, dtype = bool)))  # spam: 1, ham: 0
	# C = 0 # initial model complexity
	# LC = 0
	Para_Spam = list(Para_Spam)
	Para_Ham = list(Para_Ham)
	# Para_Spam[4] = np.ones((Para_Spam[0], D))
	# Para_Ham[4] = np.ones((Para_Ham[0], D))
	# Para_Spam = shuffle(Para_Spam)
	# Para_Ham = shuffle(Para_Ham)
	# Comp_stat = [Para_Spam[0], Para_Ham[0], 0, 0]  # M_spam, M_ham, spam_removed, ham_removed

	# 2. hard assignment and parameter re-estimation
	print("hard assignment and component specific parameter re-estimation")
	print(Para_Spam[0], Para_Ham[0])
	print(Para_Spam[1].max(), Para_Spam[1].min())
	print(Para_Ham[1].max(), Para_Ham[1].min())
	# tmp_old = np.logical_or(np.logical_and(s,np.logical_not(w_s)), np.logical_and(np.logical_not(s),w_h))
	# tmp = T => spam
	ind_spam = np.where(s == True)[0] # global index of spam
	ind_ham = np.where(s == False)[0] # global index of ham
	SpamClusters = comp_log_likelihood(Train[ind_spam], Para_Spam).argmax(axis=1)
	HamClusters = comp_log_likelihood(Train[ind_ham], Para_Ham).argmax(axis=1)
	# SpamClusters = origin_Prob(Train[ind_spam], Para_Spam).argmax(axis=1)  # (NS,1)
	# HamClusters = origin_Prob(Train[ind_ham], Para_Ham).argmax(axis=1)  # (NH,1)
	for i in range(1):
		X_s = [] # clusters of spam samples
		X_h = [] # clusters of ham samples
		ind_del = []
		for j in range(Para_Spam[0]):
			ind = np.where(SpamClusters == j)[0]
			if ind.size != 0:
				X_s.append(ind_spam[ind])
				Para_Spam[2][j,:] = Train[ind_spam[ind],:].sum(axis=0) / Train[ind_spam[ind],:].sum() # component specific parameter re-estimation
			else:
				ind_del.append(j)
		Para_Spam = list(prune_component(Para_Spam, np.asarray(ind_del, dtype = int)))
		Para_Spam[1] = np.array([samples.size for samples in X_s])/sum([samples.size for samples in X_s])
		
		ind_del = []
		for j in range(Para_Ham[0]):
			ind = np.where(HamClusters == j)[0]
			if ind.size != 0:
				X_h.append(ind_ham[ind])
				Para_Ham[2][j,:] = Train[ind_ham[ind],:].sum(axis=0) / Train[ind_ham[ind],:].sum() # component specific parameter re-estimation
			else:
				ind_del.append(j)
		Para_Ham = list(prune_component(Para_Ham, np.asarray(ind_del, dtype = int)))
		Para_Ham[1] = np.array([samples.size for samples in X_h])/sum([samples.size for samples in X_h])

		SpamClusters_new = comp_log_likelihood(Train[ind_spam], Para_Spam).argmax(axis=1)
		HamClusters_new = comp_log_likelihood(Train[ind_ham], Para_Ham).argmax(axis=1)
		# SpamClusters_new = origin_Prob(Train[ind_spam], Para_Spam).argmax(axis=1)  # (NS,1)
		# HamClusters_new = origin_Prob(Train[ind_ham], Para_Ham).argmax(axis=1)  # (NH,1)
		if np.array_equal(SpamClusters_new, SpamClusters) and np.array_equal(HamClusters, HamClusters_new):
			break
		else:
			SpamClusters = SpamClusters_new
			HamClusters = HamClusters_new

	print(Para_Spam[1].max(), Para_Spam[1].min())
	print(Para_Ham[1].max(), Para_Ham[1].min())
	Count = Para_Ham[0] + Para_Spam[0]
	# 3. compute BIC
	# old_BIC = total_BIC(Train, Para_Spam, Para_Ham, X_s, X_h)

	print("start detection")
	#-----------------phase 1------------------------------------
	# print('phase 1')
	# Para_Spam, Para_Ham, X_s, X_h, w_s, w_h = sanitization(Para_Spam, Para_Ham, Train, X_s, X_h, w_s, w_h, s, 1, 1, Attacks)
	#-----------------phase 2------------------------------------
	print('phase 2')
	Para_Spam, Para_Ham, X_s, X_h, w_s, w_h = sanitization(Para_Spam, Para_Ham, Train, X_s, X_h, w_s, w_h, s, 2, 1, Attacks)
	
	attack = np.logical_or(np.logical_and(s, w_s), np.logical_and(np.logical_not(s), w_h))
	print(attack.sum())
	# print(np.logical_xor(attack_old, attack).sum())

	return attack, Para_Ham, Para_Spam

def main(dataset, classifier, NAttackHam = 0, NAttackSpam = 0, Syn = False):

	# ---data
	class0 = {'full': 'Ham', 'AmazonReview': 'Pos'}
	class1 = {'full': 'Spam', 'AmazonReview': 'Neg'}
	if NAttackHam == 0 and NAttackSpam == 0:  # pure
		paraHamPath = 'learnedPara/'+dataset+'/learnedPara{}/'.format(class0[dataset])
		paraSpamPath = 'learnedPara/'+dataset+'/learnedPara{}/'.format(class1[dataset])
		featuresHamPath = 'dataset/'+dataset+'/features{}Train_pure.npz'.format(class0[dataset])
		featuresSpamPath = 'dataset/'+dataset+'/features{}Train_pure.npz'.format(class1[dataset])
		TestHamPath = 'dataset/'+dataset+'/features{}Test_pure.npz'.format(class0[dataset])
		TestSpamPath = 'dataset/'+dataset+'/features{}Test_pure.npz'.format(class1[dataset])
	else: # poisoned
		paraHamPath = 'learnedPara/'+dataset+'/learnedPara{}_Attack_{}_{}/'.format(class0[dataset], NAttackHam, NAttackSpam)
		paraSpamPath = 'learnedPara/'+dataset+'/learnedPara{}_Attack_{}_{}/'.format(class1[dataset], NAttackHam, NAttackSpam)
		featuresHamPath = 'dataset/'+dataset+'/features{}Train_Attack_{}_{}.npz'.format(class0[dataset], NAttackHam, NAttackSpam)
		featuresSpamPath = 'dataset/'+dataset+'/features{}Train_Attack_{}_{}.npz'.format(class1[dataset], NAttackHam, NAttackSpam)
		TestHamPath = 'dataset/'+dataset+'/features{}Test_Attack_{}_{}.npz'.format(class0[dataset], NAttackHam, NAttackSpam)
		TestSpamPath = 'dataset/'+dataset+'/features{}Test_Attack_{}_{}.npz'.format(class1[dataset], NAttackHam, NAttackSpam)
		HamGTlabelsPath = 'dataset/'+dataset+'/{}GroundTruthLabels_Attack_{}_{}.npy'.format(class0[dataset], NAttackHam, NAttackSpam)
		SpamGTlabelsPath = 'dataset/'+dataset+'/{}GroundTruthLabels_Attack_{}_{}.npy'.format(class1[dataset], NAttackHam, NAttackSpam)
		
	# load training & test data
	TrainHam = sparse.load_npz(featuresHamPath)
	TrainSpam = sparse.load_npz(featuresSpamPath)
	Train = sparse.vstack([TrainHam, TrainSpam])
	TestHam = sparse.load_npz(TestHamPath)
	TestSpam = sparse.load_npz(TestSpamPath)
	Test = sparse.vstack([TestHam, TestSpam])
	labelsTrain = np.concatenate((np.zeros(TrainHam.shape[0]), np.ones(TrainSpam.shape[0])))
	labelsTest = np.concatenate((np.zeros(TestHam.shape[0],), np.ones(TestSpam.shape[0],)))
	HamAttack = np.ones(TrainHam.shape[0], dtype = bool)
	HamAttack[:CleanHam] = False
	SpamAttack = np.ones(TrainSpam.shape[0], dtype = bool)
	SpamAttack[:CleanSpam] = False
	Attacks = np.concatenate((HamAttack, SpamAttack))
	df_poisoned = pd.read_csv('dataset/full/Attack_{}_{}_train.csv'.format(NAttackHam, NAttackSpam))
	if (Train.shape[0] != df_poisoned.shape[0]):
		print('data set size does not match')
		return
			
	# ham para
	alpha_Ham = np.load(paraHamPath+'alpha.npy')
	comp_spec_Prob_Ham = np.load(paraHamPath+'comp_spec_Prob.npy')
	shared_Prob_Ham = np.load(paraHamPath+'shared_Prob.npy')
	V_Ham = np.load(paraHamPath+'v.npy')
	M_Ham, D = V_Ham.shape
	Para_Ham = (M_Ham, alpha_Ham, comp_spec_Prob_Ham, shared_Prob_Ham, V_Ham) 

	# spam para
	alpha_Spam = np.load(paraSpamPath+'alpha.npy')
	comp_spec_Prob_Spam = np.load(paraSpamPath+'comp_spec_Prob.npy')
	shared_Prob_Spam = np.load(paraSpamPath+'shared_Prob.npy')
	V_Spam = np.load(paraSpamPath+'v.npy')
	M_Spam, D = V_Spam.shape
	Para_Spam = (M_Spam, alpha_Spam, comp_spec_Prob_Spam, shared_Prob_Spam, V_Spam)

	if NAttackSpam == 0 and NAttackHam == 0:
		spam_removed = [16]
		ham_removed = [5,6,10] # [3,7,18,19]
		SpamClusters = origin_Prob(TestSpam, Para_Spam).argmax(axis=1)  # (NS,1)
		HamClusters = origin_Prob(TestHam, Para_Ham).argmax(axis=1)  # (NH,1)

		TestSpam_in_removed_comp = []
		TestHam_in_removed_comp = []
		for j in spam_removed:
			ind = np.where(SpamClusters==j)[0]
			TestSpam_in_removed_comp += list(ind)
		for j in ham_removed:
			ind = np.where(HamClusters==j)[0]
			TestHam_in_removed_comp += list(ind)
		TestSpam_in_removed_comp = np.array(TestSpam_in_removed_comp)
		TestHam_in_removed_comp = np.array(TestHam_in_removed_comp)

	print(Train.shape)

	# avg test acc
	poisoned_acc_SVM = 0
	poisoned_acc_LR = 0
	test_acc_SVM = 0
	test_acc_LR = 0
	avg_tpr = 0
	avg_fpr = 0

	print("poisoned {} classifier".format(classifier))
	
	# -------------------BIC-based defense--------------------------------
	A, Para_Ham, Para_Spam = detection_one_at_a_time(TrainHam.shape[0], TrainSpam.shape[0], Train, Para_Ham, Para_Spam, Attacks)
	# remove suspicious samples
	TrueAttack = A*Attacks
	FalseAttack = A*np.logical_not(Attacks)
	print(TrueAttack[:TrainHam.shape[0]].sum(), TrueAttack[TrainHam.shape[0]:].sum())
	print(FalseAttack[:TrainHam.shape[0]].sum(), FalseAttack[TrainHam.shape[0]:].sum())
	if NAttackHam != 0 or NAttackSpam != 0:
		tpr = TrueAttack.sum()/(Train.shape[0]-(CleanHam+CleanSpam))
	else:
		tpr = 0
	fpr = FalseAttack.sum()/(CleanHam+CleanSpam)
	print("tpr: {}, fpr: {}".format(tpr, fpr))
	avg_tpr += tpr
	avg_fpr += fpr
	test_acc_SVM += classifiers(Train[np.logical_not(A)], labelsTrain[np.logical_not(A)], Test, labelsTest, 'SVM')
	# test_acc_LR +=classifiers(Train[np.logical_not(A)], labelsTrain[np.logical_not(A)], Test, labelsTest, 'LR')
	# df_sanitized = df_poisoned[np.logical_not(A)]
	# df_sanitized.to_csv('dataset/full/Sanitized_{}_{}_train.csv'.format(NAttackHam, NAttackSpam), index=False)
	# LSTM_classification('Sanitized_{}_{}_train.csv'.format(NAttackHam, NAttackSpam), 'test.csv')
	if NAttackSpam == 0 and NAttackHam == 0:
		print('training')
		classifiers(Train[np.logical_not(A)], labelsTrain[np.logical_not(A)], Train[A], labelsTrain[A], 'SVM')
		# avg_likelihood_in_spam = mixture_log_likelihood(Train[A], Para_Spam).sum().item()/A.sum()
		# avg_likelihood_in_ham = mixture_log_likelihood(Train[A], Para_Ham).sum().item()/A.sum()
		avg_likelihood_in_spam = comp_log_likelihood(Train[A], Para_Spam).max(axis=1).sum()/A.sum()
		avg_likelihood_in_ham = comp_log_likelihood(Train[A], Para_Ham).max(axis=1).sum()/A.sum()
		print(avg_likelihood_in_ham, avg_likelihood_in_spam)
		print('test')
		Test_in_removed_comp = sparse.vstack([TestHam[TestHam_in_removed_comp], TestSpam[TestSpam_in_removed_comp]])
		labelsTest_in_removed_comp = np.concatenate((np.zeros(TestHam_in_removed_comp.size,), np.ones(TestSpam_in_removed_comp.size,)))
		classifiers(Train[np.logical_not(A)], labelsTrain[np.logical_not(A)], Test_in_removed_comp, labelsTest_in_removed_comp, 'SVM')
	# dict_attack = {'A': Attacks, 'a': A}      
	# df_attack = pd.DataFrame(dict_attack) 
	# df_attack.to_csv('outputs/Attack_{}_{}.csv'.format(NAttackHam, NAttackSpam), index=False)
	
	'''
	# -----------------------KNN_detection-------------------------
	labels_sanitized, A = KNN_detection(Train, labelsTrain, 10)
	labels_sanitized = labels_sanitized.astype(int)
	# df_sanitized = df_poisoned[np.logical_not(A)]
	# df_sanitized.to_csv('dataset/full/Sanitized_KNN_{}_{}_train.csv'.format(NAttackHam, NAttackSpam), index=False)
	
	# df_sanitized = {'label': labels_sanitized, 'text': df_poisoned['text']}      
	# df_sanitized = pd.DataFrame(df_sanitized) 
	# df_sanitized.to_csv('dataset/full/Sanitized_KNN_{}_{}_train.csv'.format(NAttackHam, NAttackSpam), index=False)
	# test_acc_SVM += classifiers(Train[np.logical_not(A)], labelsTrain[np.logical_not(A)], Test, labelsTest, 'SVM')
	# test_acc_LR += classifiers(Train[np.logical_not(A)], labelsTrain[np.logical_not(A)], Test, labelsTest, 'LR')
	test_acc_SVM += classifiers(Train, labels_sanitized, Test, labelsTest, 'SVM')
	test_acc_LR += classifiers(Train, labels_sanitized, Test, labelsTest, 'LR')

	# fpr & tpr
	TrueAttack = A*Attacks
	FalseAttack = A*np.logical_not(Attacks)
	print(TrueAttack[:TrainHam.shape[0]].sum(), TrueAttack[TrainHam.shape[0]:].sum())
	print(FalseAttack[:TrainHam.shape[0]].sum(), FalseAttack[TrainHam.shape[0]:].sum())
	if NAttackHam != 0 or NAttackSpam != 0:
		tpr = TrueAttack.sum()/(Train.shape[0]-(CleanHam+CleanSpam))
	else:
		tpr = 0
	fpr = FalseAttack.sum()/(CleanHam+CleanSpam)
	print("tpr: {}, fpr: {}".format(tpr, fpr))

	avg_tpr += tpr
	avg_fpr += fpr
	'''
	'''
	# ----------------gradient shaping------------------------------
	test_acc_SVM += Gradient_Shaping(Train, labelsTrain, Test, labelsTest, 'SVM')
	test_acc_LR += Gradient_Shaping(Train, labelsTrain, Test, labelsTest, 'LR', epoch=20, lr=0.05)
	'''

	poisoned_acc_SVM = poisoned_acc_SVM/folder
	poisoned_acc_LR = poisoned_acc_LR/folder
	test_acc_SVM = test_acc_SVM/folder
	test_acc_LR = test_acc_LR/folder
	avg_tpr = avg_tpr/folder
	avg_fpr = avg_fpr/folder
	
	print('avg tpr {}, avg fpr {}'.format(avg_tpr, avg_fpr))
	print("avg acc of poisoned SVM {}, avg acc of retrained SVM {}".format(poisoned_acc_SVM, test_acc_SVM))
	print("avg acc of poisoned LR {}, avg acc of retrained LR {}".format(poisoned_acc_LR, test_acc_LR))

	return

if __name__ == "__main__":

	# main('full', classifier = 'LR', NAttackHam = 0, NAttackSpam = 6000)
	main('full', classifier = 'SVM', NAttackHam = 0, NAttackSpam = 2000)
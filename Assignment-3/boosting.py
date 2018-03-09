import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO: implement "predict"
		########################################################
		l=len(self.clfs_picked)
		# print(self.betas)
		# print(self.clfs_picked)
		# hx=[self.betas[j]*self.clfs_picked[j].predict(features) for j in range(l)]
		# h=[-1 if i<0 else 1 for i in hx]
		sum2=[0 for i in features]
		for i in range(0,len(self.clfs_picked)):
			temp=self.clfs_picked[i].predict(features)
			a=[self.betas[i]*j for j in temp]
			# print(a)
			sum1=[x+y for x,y in zip(a,sum2)]
			# print(sum1)
			sum2=sum1[:]
			# print(sum2)
		# print(sum2)
		h=[]
		for i in sum2:
			if(i<=0):
				h.append(-1)
			else:
				h.append(1)
		return h

		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		w0=np.ones(len(labels))/len(labels)
		h=[]
		z=[]
		self.T+=1

		for i in range(self.T):
			sum1=[]
			for j in range(0,len(self.clfs)):
				l=list(self.clfs)[j].predict(features)
				sumtemp=0
				for k in range(0,len(l)):
					if(labels[k]!=l[k]):
						sumtemp+=w0[k]
				sum1.append(sumtemp)
			ht=list(self.clfs)[np.argmin(sum1)]
			temp=ht.predict(features)
			error=0
			for j in range(0,len(labels)):
				if(labels[j]!=temp[j]):
					error+=w0[j]
			beta=0.5*np.log2((1-error)/error)
			w1=[]
			for j in range(0,len(w0)):
				if(labels[j]==temp[j]):
					w1.append(w0[j]*np.exp(-1*beta))
				else:
					w1.append(w0[j] * np.exp(beta))
			w0=[j/sum(w1) for j in w1]
			self.clfs_picked.append(ht)
			self.betas.append(beta)


	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		pi=[0.5 for i in labels]
		beta=[0 for i in features]
		self.T+=10
		for i in range(0,self.T):
			z=[]
			for j in range(0,len(labels)):
				z.append((((labels[j]+1)/2)-pi[j])/(pi[j]*(1-pi[j])))
			w=[j*(1-j) for j in pi]
			sum1 = []
			for j in range(0, len(self.clfs)):
				l = list(self.clfs)[j].predict(features)
				stemp=0
				for k in range(0,len(l)):
					stemp+=w[k]*((z[k]-l[k])**2)
				sum1.append(stemp)
			ht = list(self.clfs)[np.argmin(sum1)]
			temp=ht.predict(features)

			self.clfs_picked.append(ht)
			temp = ht.predict(features)
			b0=[]
			for k in range(0,len(temp)):
				b0.append(beta[k]+temp[k])
			beta=b0[:]
			pi1=[]
			for k in range(0,len(pi)):
				pi1.append(1/(1+np.exp(-2*beta[k])))
			pi=pi1[:]
		self.betas=[0.5 for i in self.clfs_picked]






		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	
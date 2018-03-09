import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		# print(features)
		# print(labels)
		# print(num_cls)
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True
		if len(self.features) ==0:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of
					  Corresponding training samples
					  eg:[[2,2],[4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			bran=np.array(branches)
			eleBranch=np.sum(bran,axis=0)
			noClasses=len(branches)
			noBranches=len(branches[0])
			totalattr=np.sum(eleBranch)
			val=[]
			for i in range(0,noBranches):
				sum=0
				for j in range(0,noClasses):
					if(branches[j][i]==0):
						continue
					sum+=(branches[j][i]/eleBranch[i])*np.log2(branches[j][i]/eleBranch[i])
				val.append(sum)
			value=0
			for i in range(0,len(eleBranch)):
				value+=(eleBranch[i]/totalattr)*-1*val[i]
			return value



		fet={}
		for idx_dim in range(len(self.features[0])):
			############################################################
			# TODO: compare each split using conditional entropy
			#       find the best split
			############################################################
			# print("hai",idx_dim)
			label=self.labels[:]
			feat = np.array(self.features)[:,idx_dim]
			keys=np.unique(np.array(feat))
			# print(keys)
			labuniq=np.unique(self.labels)
			ret=[]
			for i in range(len(labuniq)):
				ret.append([])
			dim=len(keys)
			for i in keys.tolist():
				indices = np.where(feat==i)[0]
				lab=[]
				for j in indices:
					lab.append(label[j])
					# label.remove(j)
				unique, counts = np.unique(lab, return_counts=True)
				for k in range(len(ret)):
					if( k in unique):

						ret[k].append(counts[int(np.where(unique==k)[0])])
					else:
						ret[k].append(0)
				# print(ret)
			fet[idx_dim]=conditional_entropy(ret)
		# print(fet)




		key=0
		max=0
		for i in fet:
			if(fet[i]>max):
				max=fet[i]
				key=i
		self.dim_split = key

		# for i in range(len(self.features[0])):
		feat = np.array(self.features)[:, key]
		keys = np.unique(np.array(feat))
		abc=[]
		# self.feature_uniq_split=keys.tolist()[:]
		for m in keys:
			f1 = []
			lab=[]
			for j in range(0,len(self.features)):
				inn=[]
				# for k in range(0,len(self.features[0])):
				if(self.features[j][self.dim_split]==m):
					inn=np.array(self.features[j]).tolist()
					lab.append(self.labels[j])
					inn.remove(m)
					if(len(inn)==0):
						continue
					f1.append(inn)

			len1=len(np.unique(np.array(lab)))
			# if(len(f1)==0):
				# continue
			abc.append(m)

			self.children.append(TreeNode(features=f1,labels=lab,num_cls=len1))
			# print(m)
			# print(f1)
			# print(lab)
		self.feature_uniq_split=abc[:]














	############################################################
		# TODO: split the node, add child nodes
		############################################################




		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			# print(self.dim_split)
			# print(self.feature_uniq_split)
			if(feature[self.dim_split] not in self.feature_uniq_split ):
				return self.cls_max
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			# print(idx_child)
			# print(self.children)
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max




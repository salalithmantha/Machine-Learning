from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.train_features=features
        self.train_labels=labels
        #print("labels",labels)


    def predict(self, features: List[List[float]]) -> List[int]:
        labels=[]
        for i in range(0,len(features)):
            mapdist = {}
            labelzero = []
            labelone = []
            for j in range(0,len(self.train_features)):
                dist=self.distance_function(features[i],self.train_features[j])
                mapdist[dist]=self.train_labels[j]
            key1=list(mapdist.keys())
            keys=sorted(key1)
            if(self.k<=len(keys)):
                for p in range(0,self.k):
                    t=keys[p]
                    if(mapdist.get(t)==0):
                        labelzero.append(0)
                    elif(mapdist.get(t)==1):
                        labelone.append(1)
            if(len(labelone)>len(labelzero)):
                labels.append(1)
            else:
                labels.append(0)
        # print(keys)
        # print(labels)
        return labels







if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)

from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        self.w[0]=1
        pyW=np.matrix(self.w)

        for i in range(0,self.max_iteration):
            l1=[]
            l2=[]

            arr = np.arange(len(features))
            #print(len(features))
            np.random.shuffle(arr)
            for km in arr.tolist():
                l1.append(features[km])
                l2.append(labels[km])



            #random.shuffle(features)
            for z in range(0,len(l1)):
                #z=random.randint(0,len(features)-1)
                x=l1[z]
                #print(i)

                y=pyW*np.transpose(np.matrix(x))


                if(y.tolist()[0][0]*l2[z]<self.margin):
                    x1=[km*l2[z] for km in x]
                    modx=np.linalg.norm(np.array(x))
                    x2=[km/modx for km in x1]

                    pyW=pyW+np.matrix(x2)
            self.w= pyW.tolist()[0]
            #print(self.w)


            for j in range(0,len(features)):
                sum=0
                #print(pyW)
                #print(features[j])
                y1=pyW*np.transpose(np.matrix(l1[j]))
                if(y1.tolist()[0][0]*l2[j]<self.margin):
                    sum+=1
                if(sum==0):

                    #print("true")
                    return True









    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        label=[]
        for i in range(0, len(features)):
            z=self.w

            y=np.matrix(z)*np.transpose(np.matrix(features[i]))
            #print(y)
            kmt=y.tolist()[0][0]
            #print(kmt)
            if(kmt<0):
                label.append(-1)
            else:
                label.append(1)
        return label





    def get_weights(self) -> Tuple[List[float], float]:
        return self.w

#model=Perceptron(2)
#k=model.train([[1,2,3],[1,2,3]],[-1,-1])
#print(k)

#k=model.predict([[1,2,3],[2,3,4]])
#print(k)
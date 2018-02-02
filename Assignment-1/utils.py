from typing import List

import numpy as np



def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    sum=0

    for i in range(0,len(y_pred)):
        x=(y_pred[i]-y_true[i])**2
        sum+=x
    return sum/len(y_pred)


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    c_p=0
    a_p_p=0
    a_p_r=0
    for i in range(0,len(real_labels)):
        if(predicted_labels[i]==1):
            if(real_labels[i]==1):
                c_p+=1
        if(real_labels[i]==1):
            a_p_r+=1
        if(predicted_labels[i]==1):
            a_p_p+=1
    p=c_p/a_p_p
    r=c_p/a_p_r
    if(c_p==0):
        return 0
    f=2*((p*r)/(p+r))
    return f





def polynomial_features(features: List[List[float]], k: int) -> List[List[float]]:
    #d=features[:]
    out=[]
    for inner in features:
        d=[]
        for i in range(1,k+1):
            temp=[j**i for j in inner]
            d=d+temp
        out.append(d)
    return out



def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    sum=0
    for i in range(0,len(point1)):
        x=(point1[i]-point2[i])**2
        sum+=x
    return sum**(1/2)


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    sum=0
    for i in range(0,len(point2)):
        x=point2[i]*point1[i]
        sum+=x
    return sum



def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:

    return np.exp((-0.5)*euclidean_distance(point1,point2))



def normalize(features: List[List[float]]) -> List[List[float]]:
    """
    normalize the feature vector for each sample . For example,
    if the input features = [[3, 4], [1, -1], [0, 0]],
    the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
    """
    nor=[]
    for i in features:
        sum=0
        inner=[]
        for j in i:
            sum+=j**2
        den=sum**(1/2)
        for j in i:
            if(sum==0):
                inner.append(0)
            else:
                inner.append(j/den)
        nor.append(inner)
    return nor



def min_max_scale(features: List[List[float]]) -> List[List[float]]:
    """
    normalize the feature vector for each sample . For example,
    if the input features = [[2, -1], [-1, 5], [0, 0]],
    the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
    """
    k=np.array(features,dtype=float)
    max=np.amax(k,axis=0)
    min=np.amin(k,axis=0)
    for i in range(0,len(features)):
        for j in range(0,len(features[i])):
            k[i][j]=(k[i][j]-min[j])/(max[j]-min[j])

    return k.tolist()



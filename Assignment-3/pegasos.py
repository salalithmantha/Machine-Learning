import json
import numpy as np


###### Q1.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    Return:
    - train_obj: the value of objective function in SVM primal formulation
    """
    # you need to fill in your solution here
    xList = np.array(X).tolist()
    train = []
    for i in xList:
        z = [1] + i
        train.append(z)
    # print(X)
    z = np.matrix(train)
    w1 = np.transpose(np.matrix(w))
    sum=0

    for i in range(0, len(y)):
        k = 1 - y[i] * np.matmul(np.transpose(w1), np.transpose(np.matrix(z[i]))).tolist()[0][0]
        m = max(0, k)
        sum += m
    sum = sum / len(y)
    wsqr = np.matmul(w1, np.transpose(w1))
    wsqr = lamb * wsqr / 2
    obj_value = wsqr + sum

    return obj_value.tolist()[0]






###### Q1.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the maximum number of iterations to update parameters
    Returns:
    - learnt w
    - traiin_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]
    w1 = np.transpose(np.matrix(np.append(0, w)))
    xlist = Xtrain.tolist()
    train = []
    for i in xlist:
        z = [1] + i
        train.append(z)
    z = np.matrix(train)



    train_obj = []

    for iter in range(1, max_iterations + 1):
        A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch

        # you need to fill in your solution here
        aplus=[]
        alist=[i for i in range(0,N)]
        for i in A_t:
            # print(np.shape(np.transpose(w1)))
            # print(np.shape(np.transpose(z[i])))
            # yy=np.random.random_integers(0,N)
            # while(yy not in alist):
            #     yy=np.random.random_integers(0,N)
            f=ytrain[i]*np.matmul(np.transpose(w1),np.transpose(z[i])).tolist()[0][0]
            # print(f)
            if(f<1):
                aplus.append(i)
        insum = [0 for j in range(0, D + 1)]
        insum = np.transpose(np.matrix(insum))
        # print(insum)
        # print(np.shape(w1))
        for kk in aplus:

            # print(np.shape(np.transpose(z[k])*ytrain[k]))
            insum = np.add(insum,(np.transpose(z[kk])*ytrain[kk]))
            # print(np.shape(insum))
        nu = 1 / (lamb * iter)
        # print(k)
        w15 = np.add((1 - (nu * lamb)) * w1,((nu)/(k))*insum)
        # print(np.sqrt(lamb))
        # if(np.linalg.norm(w15)!=0):
        w2 = min(1, ((1 / (np.sqrt(lamb)*np.linalg.norm(w15)))))* w15
        # else:
        #     w2=np.copy(w15)
        w1 = np.copy(w2)
        znow=[]
        ynow=[]
        for i in A_t:
            znow.append(Xtrain[i])
            ynow.append(ytrain[i])
        w = np.transpose(np.copy(w1)).tolist()[0]
        train_obj.append(objective_function(znow,ynow,w,lamb))



    w=np.transpose(np.copy(w1)).tolist()[0]
    return w, train_obj


###### Q1.3 ######
def pegasos_test(Xtest, ytest, w, t = 0.):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
    - t: threshold, when you get the prediction from SVM classifier, it should be real number from -1 to 1. Make all prediction less than t to -1 and otherwise make to 1 (Binarize)
    Returns:
    - test_acc: testing accuracy.
    """
    # you need to fill in your solution here
    f = []
    correct = 0
    wrong = 0
    # xList = np.array(Xtest).tolist()
    # train = []
    # for i in xList:
    #     z = [1] + i
    #     train.append(z)
    # print(X)
    z = np.matrix(Xtest)

    for i in range(0, len(ytest)):
        f.append(w[0]+np.matmul(np.matrix(w[1:]), np.transpose(z[i])).tolist()[0][0])
    # print(f)
    # print(ytest)
    # print(f)
    for i in range(0, len(ytest)):
        if (f[i] < t):
            if (ytest[i] == -1):
                correct += 1
        else:
            if (ytest[i] == 1):
                correct += 1
    test_acc = correct / len(ytest)

    return test_acc


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')

    max_iterations = 500
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist() # results on mnist
    print('mnist test acc \n')
    for key, value in test_acc.items():
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
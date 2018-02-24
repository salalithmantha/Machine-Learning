from __future__ import division, print_function

import numpy as np
import scipy as sp

# from matplotlib import pyplot as plt
# from matplotlib import cm


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def binary_train(X, y, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - step_size: step size (learning rate)
    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic regression
    - b: scalar, which is the bias of logistic regression
    Find the optimal parameters w and b for inputs X and y.
    Use the average of the gradients for all training examples to
    update parameters.
    """

    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0
    w2 = np.matrix(np.append(b, w))

    xList = np.array(X).tolist()
    train = []
    for i in xList:
        z = [1] + i
        train.append(z)
    z = np.matrix(train)
    tempW = np.matrix(w2.tolist()[0])
    max2=0
    maxw = np.matrix(w2.tolist()[0])

    for step in range(0,max_iterations):
        l_x = []
        l_y = []

        arr = np.arange(len(X))
        np.random.shuffle(arr)
        for km in arr.tolist():
            l_x.append(z[km])
            l_y.append(y[km])



        for i in range(0, len(l_x)):

            score = np.dot(tempW, np.transpose(l_x[i]))

            pred = step_size * np.dot((1 / (1 + np.exp(-score))) - l_y[i], np.matrix(l_x[i]))
            tempW = tempW - pred

        sum = 0
        for i in range(0, len(l_x)):
            score = 1 / (1 + np.exp(-(np.dot(tempW, np.transpose(l_x[i])))))[0]
            if (score > 0.5):
                k = 1
            elif (score < 0.5):
                k = 0
            if (l_y[i] != k):
                sum += 1
        # print(1-sum/len(l_x))

        if(1-sum/len(l_x)>max2):
            max2=1 - sum / len(X)

            maxw=np.matrix(tempW)
        if(step>100):
            w = np.array(maxw.tolist()[0][1:])
            b = maxw.tolist()[0][0]
            # print("____________________________________________________________________")
            return w, b


        if (sum == 0):
            w = np.array(tempW.tolist()[0][1:])
            b = tempW.tolist()[0][0]
            return w, b







    """
    TODO: add your code here
    """
    w = np.array(tempW.tolist()[0][1:])
    b = tempW.tolist()[0][0]



    assert w.shape == (D,)
    return w, b


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    # print(X)
    N, D = X.shape
    preds = np.zeros(N)
    z = np.matrix(X)
    y = []
    # print(X)
    for i in range(0, len(X)):
        score = 1 / (1 + np.exp(-(b + np.dot(w, np.transpose(z[i])))))[0]
        # print(score)

        if (score > 0.5):
            y.append(1)
        elif (score < 0.5):
            y.append(0)
    preds = np.array(y)

    """
    TODO: add your code here
    """
    assert preds.shape == (N,)
    return preds


def multinomial_train(X, y, C,
                      w0=None,
                      b0=None,
                      step_size=0.5,
                      max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - step_size: step size (learning rate)
    - max_iterations: maximum number for iterations to perform
    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    Implement a multinomial logistic regression for multiclass
    classification. Keep in mind, that for this task you may need a
    special (one-hot) representation of classification labels, where
    each label y_i is represented as a row of zeros with a single 1 in
    the column, that corresponds to the class y_i belongs to.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    # print(w)
    # print(b)
    w_00 = []
    for i in range(0, len(w)):
        w_00.append(np.append(b[i], w[i]))
    wk = np.matrix(w_00)

    # print(wk)



    xList = np.array(X).tolist()
    train = []
    for i in xList:
        z = [1] + i
        train.append(z)
    z = np.matrix(train)
    pyW = np.matrix(wk)

    y_00 = []
    for i in range(0, len(y)):
        y_o = [0 for j in range(0, C)]
        y_o[y[i]] = 1
        y_00.append(y_o)
    # print(y_00)


    max2=0
    maxw = np.matrix(wk)

    for step in range(0, max_iterations):
        l1 = []
        l2 = []

        arr = np.arange(len(X))
        # np.random.shuffle(arr)
        for km in arr.tolist():
            l1.append(z[km])
            l2.append(y_00[km])

        for i in range(0, len(l1)):

            ytm = retl(l1[i], pyW, 1)
            for k in range(0, C):


                pred = step_size * np.dot(ytm[k] - l2[i][k], np.matrix(l1[i]))
                # print(pred)
                pyW[k] = pyW[k] - pred
                # print(pyW)

        sum = 0
        for i in range(0, len(l1)):
            max1 = []
            k = retl(l1[i], pyW, 0)
            if (y[i] != k):
                sum += 1

        # print(1-sum/len(l1))

        if (1 - sum / len(X) > max2):
            max2=1 - sum / len(X)
            maxw = np.matrix(pyW).tolist()
        if (step > 25):
            w3=[]
            b3=[]
            for k32 in maxw:
                b3.append(k32[0])
                k = k32[1:]
                w3.append(k)
            w4 = np.array(w3)
            b4 = np.array(b3)
            # print("- - - - - -- - - - -- - - - - -- - - - -")

            assert w4.shape == (C, D)
            assert b4.shape == (C,)
            return w4,b4

        if (sum == 0):
            # print("Hello")
            w = []
            b = []
            w1 = pyW.tolist()
            # print(w1)
            for i in w1:
                b.append(i[0])
                k = i[1:]
                w.append(k)
            w = np.array(w)
            b = np.array(b)
            return w, b

    w = []
    b = []
    w1 = pyW.tolist()
    # print(w1)
    for i in w1:
        b.append(i[0])
        k = i[1:]
        w.append(k)
    w = np.array(w)
    b = np.array(b)

    """
    TODO: add your code here
    """

    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b


def retl(x, w, sel):
    yy = []
    for i in range(0, len(w)):
        scores = np.dot(w[i], np.transpose(x))
        yy.append(scores)
    k = softMax(yy)
    sol = []
    for i in range(0, len(w)):
        z = k[i] / sum(k)
        sol.append(z)
    if (sel == 1):
        return sol
    return sol.index(max(sol))


def ret(x, w, b):
    yy = []
    for i in range(0, len(w)):
        scores = np.dot(w[i], np.transpose(x))
        yy.append(b[i] + scores)
    k = softMax(yy)
    sol = []
    for i in range(0, len(w)):
        z = k[i] / sum(k)
        sol.append(z)
    return sol.index(max(sol))


def softMax(w):
    y = []
    for j in w:
        y.append(np.exp(j))
    return y


def multinomial_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier
    - b: bias terms of the trained multinomial classifier

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    Make predictions for multinomial classifier.
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N)

    """
    TODO: add your code here
    """
    pyW = np.array(w)
    bb = np.array(b)
    l1 = np.array(X)
    z = []

    for i in range(0, len(l1)):
        max1 = []
        k = ret(l1[i], pyW, bb)
        z.append(k)
    preds = np.array(z)
    # print(preds)

    assert preds.shape == (N,)
    return preds


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array,
    indicating the labels of each training point
    - C: number of classes in the data
    - w0: initial value of weight matrix
    - b0: initial value of bias term
    - step_size: step size (learning rate)
    - max_iterations: maximum number of iterations for gradient descent
    Returns:
    - w: a C-by-D weight matrix of OVR logistic regression
    - b: bias vector of length C
    Implement multiclass classification using binary classifier and
    one-versus-rest strategy. Recall, that the OVR classifier is
    trained by training C different classifiers.
    """
    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    # print(C)
    # print(y)
    pyW = []
    bb = []
    for i in range(0, C):
        train_y = [1 if k == i else 0 for k in y]
        # print(train_y)
        l, k = binary_train(X, train_y)
        pyW.append(l)
        bb.append(k)

    w = np.array(pyW)
    b = np.array(bb)
    # print(w)
    # print(b)





    """
    TODO: add your code here
    """
    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b


def OVR_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained OVR model
    - b: bias terms of the trained OVR model

    Returns:
    - preds: vector of class label predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes.
    Make predictions using OVR strategy and predictions from binary
    classifier.
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N)

    """
    TODO: add your code here
    """
    a = []

    for i in range(0, C):
        # N,D=np.array(np.matrix(X).tolist()[0]).shape
        a.append(binary_predict(X, w[i], b[i]))
    k = np.argmax(np.array(a), 0)
    preds = np.array(k)

    assert preds.shape == (N,)
    return preds


#######################################################################
# DO NOT MODIFY THE CODE BELOW
#######################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)


def run_binary():
    from data_loader import toy_data_binary, \
        data_loader_mnist

    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()

    w, b = binary_train(X_train, y_train)

    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)

    print('train acc: %f, test acc: %f' %
          (accuracy_score(y_train, train_preds),
           accuracy_score(y_test, preds)))

    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train]
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test]

    w, b = binary_train(X_train, binarized_y_train)

    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)

    print('train acc: %f, test acc: %f' %
          (accuracy_score(binarized_y_train, train_preds),
           accuracy_score(binarized_y_test, preds)))


def run_multiclass():
    from data_loader import toy_data_multiclass_3_classes_non_separable, \
        toy_data_multiclass_5_classes, \
        data_loader_mnist

    datasets = [(toy_data_multiclass_3_classes_non_separable(),
                 'Synthetic data', 3),
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5),
                (data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data

        print('One-versus-rest:')
        w, b = OVR_train(X_train, y_train, C=num_classes)
        train_preds = OVR_predict(X_train, w=w, b=b)
        preds = OVR_predict(X_test, w=w, b=b)

        print('train acc: %f, test acc: %f' %
              (accuracy_score(y_train, train_preds),
               accuracy_score(y_test, preds)))

        print('Multinomial:')
        w, b = multinomial_train(X_train, y_train, C=num_classes)
        train_preds = multinomial_predict(X_train, w=w, b=b)
        preds = multinomial_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' %
              (accuracy_score(y_train, train_preds),
               accuracy_score(y_test, preds)))


if __name__ == '__main__':

    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()
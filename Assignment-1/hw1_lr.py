from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        f=[]

        # print(features)
        for i in range(0,len(features)):
            a=[1]+features[i]
            f.append(a)
        # print(f)



        x=numpy.matrix(f)
       # print(x)
        print()
        xt=numpy.transpose(x)
        # print(xt)
        print()
        xxt=numpy.dot(xt,x)

        xxtI=numpy.linalg.inv(xxt)

        #print(xt)
        ww=xxtI*xt
        yyy=numpy.transpose(numpy.matrix(values))

        #print(yyy)

        self.w=ww*yyy
        #print(self.w)



    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        output=[]
        # print(self.w.tolist())
        # print()

        ww=self.w.tolist()

        w0=ww[0][0]
        w1=ww[1:]
        #print(numpy.matrix(w1))
        # print()
        #print(numpy.matrix(features[0]))
        # print()

        #print(ww)

        for i in features:
            yint=numpy.transpose(numpy.matrix(w1))*numpy.transpose(numpy.matrix(i))

            y=w0+yint.tolist()[0][0]
            output.append(y)
        # print(output)



        return output



    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.w.tolist()


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        f = []

        # print(features)
        for i in range(0, len(features)):
            a = [1] + features[i]
            f.append(a)
        # print(f)



        x = numpy.matrix(f)
        # print(x)
        print()
        xt = numpy.transpose(x)
        # print(xt)
        print()
        xxt = numpy.dot(xt, x)
        r=len(xxt.tolist())
        #print(r)
        #print(xxt)
        I=self.alpha*numpy.identity(r)
        Imat=numpy.matrix(I)

        xalpha=xxt*Imat

        xxtI = numpy.linalg.inv(xalpha)

        # print(xt)
        ww = xxtI * xt
        yyy = numpy.transpose(numpy.matrix(values))

        # print(yyy)

        self.w = ww * yyy
        # print(self.w)

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        output = []
        # print(self.w.tolist())
        # print()

        ww = self.w.tolist()

        w0 = ww[0][0]
        w1 = ww[1:]
        # print(numpy.matrix(w1))
        # print()
        # print(numpy.matrix(features[0]))
        # print()

        # print(ww)

        for i in features:
            yint = numpy.transpose(numpy.matrix(w1)) * numpy.transpose(numpy.matrix(i))

            y = w0 + yint.tolist()[0][0]
            output.append(y)
        # print(output)



        return output

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.w.tolist()


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)

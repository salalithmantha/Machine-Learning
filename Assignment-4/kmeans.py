import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeans class (filename: kmeans.py')
        mean=[]
        jinit=9223372036854775807
        r=np.zeros((N,self.n_cluster),dtype=np.int)
        # print(r)
        a=np.random.choice(N,self.n_cluster,replace=False)
        # print(a)
        update=0
        for i in a:
            mean.append(x[i].tolist())
        # print(mean)
        for iter in range(0,self.max_iter):
            # print(mean)
            r = np.zeros((N, self.n_cluster), dtype=np.int)

            for i in range(0,N):
                # temp=[]
                min=9223372036854775807
                # print(i)
                minindex=0
                for j in range(0,len(mean)):
                    kt=np.sum((x[i]-mean[j])**2)
                    if(kt<min):
                        min=kt
                        minindex=j
                # l=np.argmin(temp)
                r[i][minindex]=1
            # print(r)
            #print(update)


            # for i in range(0,N):
            #     l=np.argmin([np.sum((x[i]-k)**2) for k in mean])
            #     r[i][l]=1
            # print(update)


            sum1=0
            for i in range(0,N):
                for j in range(0,self.n_cluster):
                    if(r[i][j]==1):
                        sum1+=np.linalg.norm(x[i]-mean[j])**2
            # print(sum1)
            sum1=np.round(sum1,3)
            # print(sum1)

            if(abs(jinit-sum1)<=self.e):
                break
            jinit=sum1
            numer=[0 for i in range(0,len(x[0]))]
            denom=0
            mean1=[]
            for i in range(0,self.n_cluster):
                numer=[0 for lms in range(0,len(x[0]))]

                denom=0
                for j in range(0,N):
                   denom+=r[j][i]
                   if(r[j][i]==1):
                       numer+=x[j]
                mean1.append((numer/denom).tolist())
            update+=1

            mean=[]
            for i in mean1:
                mean.append(i)
            mean=np.round(mean,3)
        R=[]
        for i in range(0,N):
            for j in range(0,self.n_cluster):
                if(r[i][j]==1):
                    R.append(j)
        # print(r)
        return (np.array(mean),np.array(R),update)







        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeansClassifier class (filename: kmeans.py')

        model=KMeans(self.n_cluster,self.max_iter,self.e)
        centroids,membership,i=model.fit(x)
        centroid_labels=[]
        for i in range(0,len(centroids)):
            temp=[]
            for j in range(0,len(membership)):
                if(membership[j]==i):
                   temp.append(y[j])
            bin=np.bincount(temp)
            centroid_labels.append(np.argmax(bin))

        centroids=np.array(centroids)
        centroid_labels=np.array(centroid_labels)
        # print(centroids)
        # print(centroid_labels)
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement predict function in KMeansClassifier class (filename: kmeans.py')

        labels=[]
        for i in range(0,N):
            min = 9223372036854775807
            val=0
            for j in range(0,len(self.centroids)):
                kt = np.sum((x[i] - self.centroids[j]) ** 2)
                if(kt<min):
                    min=kt
                    val=j
            labels.append(self.centroid_labels[val])
        # print(labels)
        return np.array(labels)
        # DONOT CHANGE CODE BELOW THIS LINE

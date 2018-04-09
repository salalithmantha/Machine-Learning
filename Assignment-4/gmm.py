import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape
        # k_means
        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k using k-means')
            model=KMeans(self.n_cluster,self.max_iter,self.e)
            mean,membership,i = model.fit(x)
            z = np.array(membership)
            y = np.bincount(z)
            pi=[i/len(membership) for i in y]
            var=[]
            for i in range(0,len(mean)):
                numer=np.zeros((D,D),dtype=float)
                for j in range(0,N):
                    if(membership[j]==i):
                        diff=x[j]-mean[i]
                        diff=np.matrix(diff)
                        # print(diff)
                        diffT=np.transpose(diff)
                        # print(np.matmul(diffT,diff))
                        numer+=np.matmul(diffT,diff)
                var.append(numer/y[i])
             # var=np.round(var,3)
            # print(var)
            # print(np.array(var).shape)
            # print(pi)








            # DONOT MODIFY CODE BELOW THIS LINE
        #    random
        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #      'Implement initialization of variances, means, pi_k randomly')
            mean = np.random.random_sample((self.n_cluster, D))
            var = []
            for i in range(0, self.n_cluster):
                var.append(np.identity(D))
            var = np.array(var)
            pi = [1 / self.n_cluster for i in range(self.n_cluster)]




            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement fit function (filename: gmm.py)')
        self.means = np.array(mean)
        self.variances=np.array(var)
        self.pi_k = pi
        l1=self.compute_log_likelihood(x)
        # step 4 shoud code
        # print(var)
        update=0
        for iter in range(0, self.max_iter):
            #print(iter)
            update+=1
            # step 6
            gamma=[]
            for i in range(0, N):
                temp=[]
                tempgamma=[]
                inexplist=[]
                for j in range(0, len(mean)):
                    diff=x[i]-mean[j]
                    diff=np.matrix(diff)
                    diffT=np.transpose(diff)
                    det=np.linalg.det(var[j])
                    if(det==0):
                        # print("hello")
                        while(det==0):
                            tempvar=var[j]+10**(-3)*np.identity(D)
                            var[j]=tempvar
                            det=np.linalg.det(var[j])
                    # inexp=-1/2*(diffT*np.linalg.inv(var[j])*diff)
                    # print(var[j])
                    # print(np.linalg.det(var[j]))
                    # print("hello")
                    # print(np.linalg.inv(np.matrix(var[j])))
                    dv=np.matmul(diff,np.linalg.inv(var[j]))
                    inexp=np.matmul(dv,diffT)/(-2)
                    # print(inexp)
                    inexplist.append(inexp.tolist()[0][0])
                # print(inexplist)
                for j in range(0,len(mean)):
                    # inexp=np.round(inexp.tolist()[0][0],3)
                    # print(inexp)
                    # outexp=np.exp((inexplist[j]-min(inexplist))/(max(inexplist)-min(inexplist)))
                    outexp=np.exp(inexplist[j])
                    norm=outexp/(2*np.pi*abs(np.linalg.det(var[j])))**(0.5)
                    # print(inexp,outexp)
                    temp.append(pi[j]*norm)
                sum1=sum(temp)
                for j in range(0,len(mean)):
                    tempgamma.append(temp[j]/sum1)
                gamma.append(tempgamma)
            # print(gamma)
            gamma=np.array(gamma)

        #     N_K calc
            N_K=[]
            for i in range(0,len(mean)):
                sum1=0
                for j in range(0,len(x)):
                    sum1+=gamma[j][i]
                N_K.append(sum1)
            # print(N_K)

        #     mean calc
            mean1=[]
            for i in range(0,len(mean)):
                sum1=0
                for j in range(0,len(x)):
                    sum1+=gamma[j][i]*x[j]
                mean1.append(np.array(sum1).tolist()/N_K[i])
            # print(mean1)
            mean=np.array(mean1)
            self.means=np.array(mean1)

        #     var calc
            var = []
            for i in range(0, len(mean)):
                numer = np.zeros((D, D), dtype=float)
                for j in range(0, N):
                    diff = x[j] - mean[i]
                    diff = np.matrix(diff)
                    # print(diff)
                    diffT = np.transpose(diff)
                    # print(np.matmul(diffT,diff))
                    numer +=gamma[j][i]*np.matmul(diffT, diff)
                var.append(numer / N_K[i])
            # print(var)
            self.variances=np.array(var)

        #     pi_k calc
            pi=[]
            for i in range(0,len(mean)):
                pi.append(N_K[i]/len(x))
            # print(pi)
            self.pi_k=pi


            l2=self.compute_log_likelihood(x)
            if(abs(l1-l2)<self.e):
                break;
            l1=l2
        self.means=np.array(mean)
        self.variances=np.array(var)
        self.pi_k=np.array(pi)
        return update












        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement sample function in gmm.py')
        K=np.random.multinomial(100,self.pi_k,1)
        # print(K)
        z=[]
        for i in range(0,len(K[0])):
            x=np.random.multivariate_normal(self.means[i], self.variances[i], K[0][i])
            for j in range(0,K[0][i]):
                z.append(np.array(x[j]).tolist())

        lt=np.array(z)
        np.random.shuffle(lt)
        return lt








        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement compute_log_likelihood function in gmm.py')
        temp = []
        mean=np.array(self.means)
        var=np.array(self.variances)
        pi=np.array(self.pi_k)
        sum1=0
        for i in range(0, len(x)):
            inexplist = []
            for j in range(0, len(mean)):
                diff = x[i] - mean[j]
                diff = np.matrix(diff)
                diffT = np.transpose(diff)
                det = np.linalg.det(var[j])
                if (det == 0):
                    # print("hello")
                    while (det == 0):
                        tempvar = var[j] + 10 ** (-3) * np.identity(len(x[0]))
                        var[j] = tempvar
                        det = np.linalg.det(var[j])

                dv = np.matmul(diff, np.linalg.inv(var[j]))
                inexp = np.matmul(dv, diffT) / (-2)
                inexplist.append(inexp.tolist()[0][0])
            temp=[]
            for j in range(0, len(mean)):
                # outexp=np.exp((inexplist[j]-min(inexplist))/(max(inexplist)-min(inexplist)))
                outexp = np.exp(inexplist[j])
                norm = outexp / (((2 * np.pi)**(len(x[0]))) * abs(np.linalg.det(var[j])))**(0.5)
                temp.append(pi[j] * norm)

            sum1+= np.log(sum(temp))
        # print(temp)
        # print(type(np.log(sum1)))
        return float(sum1)




        # DONOT MODIFY CODE BELOW THIS LINE


from __future__ import print_function
import json
import numpy as np
import sys


def forward(pi, A, B, O):
    """
    Forward algorithm

    Inputs:
    - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
    """
    S = len(pi)
    N = len(O)
    alpha = np.zeros([S, N])
    ###################################################
    # Q3.1 Edit here
    ###################################################

    for i in range(0, S):
        alpha[i][0] = pi[i] * B[i][O[0]]
        # print(alpha)

    for i in range(1, N):
        for j in range(0, S):
            sum=0
            for k in range(len(A[0])):
                sum+=alpha[k][i-1]*A[k][j]
                # print(sum)
            alpha[j][i]=B[j][O[i]]*sum
    # print(alpha)
    return alpha


def backward(pi, A, B, O):
    """
    Backward algorithm

    Inputs:
    - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
    """
    S = len(pi)
    N = len(O)
    beta = np.zeros([S, N])
    ###################################################
    # Q3.1 Edit here
    ###################################################

    for i in range(0, S):
        beta[i][N - 1] = 1

    for i in range(N-1,0,-1):
        for j in range(0, S):
            sum = 0
            for k in range(S):
                sum += beta[k][i]*A[j][k]*B[k][O[i]]
            beta[j][i-1] = sum
    return beta


def seqprob_forward(alpha):
    """
    Total probability of observing the whole sequence using the forward algorithm

    Inputs:
    - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)

    Returns:
    - prob: A float number of P(x_1:x_T)
    """
    prob = 0
    ###################################################
    # Q3.2 Edit here
    ###################################################
    for i in range(alpha.shape[0]):
        prob = prob + alpha[i, alpha.shape[1] - 1]
    return prob


def seqprob_backward(beta, pi, B, O):
    """
    Total probability of observing the whole sequence using the backward algorithm

    Inputs:
    - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
    - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence
        (in terms of the observation index, not the actual symbol)

    Returns:
    - prob: A float number of P(x_1:x_T)
    """
    prob = 0
    ###################################################
    # Q3.2 Edit here
    ###################################################
    for i in range(beta.shape[0]):
        prob=prob+beta[i,0]*pi[i]*B[i,O[0]]
    return prob


def viterbi(pi, A, B, O):
    """
    Viterbi algorithm

    Inputs:
    - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - path: A list of the most likely hidden state path k* (in terms of the state index)
      argmax_k P(s_k1:s_kT | x_1:x_T)
    """
    path = []
    ###################################################
    # Q3.3 Edit here
    ###################################################
    # " Algorithm :Referred some websites and some other textbooks"

    S=len(pi)
    N=len(O)
    # print(S)
    # print(N)
    path={}
    w = [{}]
    for i in range(0,S):
        w[0][i]=pi[i]*B[i][O[0]]
        path[i]=[i]
        # print(path)

    for i in range(1,N):
        w.append({})
        updatedpath={}
        for y in range(0,S):
            (probability,state) = max((w[i-1][k]*A[k][y]*B[y][O[i]],k) for k in range(0,S))
            w[i][y] = probability
            # print(w)
            # print(probability)
            # print(state)
            updatedpath[y] = path[state] + [y]
        path = updatedpath

    n = 0
    if(N!=1):
        n = i
    (probability,state) = max((w[n][y], y) for y in range(0,S))
    # print(path)

    return path[state]


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
    model_file = sys.argv[1]
    Osymbols = sys.argv[2]

    #### load data ####
    with open(model_file, 'r') as f:
        data = json.load(f)
    A = np.array(data['A'])
    B = np.array(data['B'])
    pi = np.array(data['pi'])
    #### observation symbols #####
    obs_symbols = data['observations']
    #### state symbols #####
    states_symbols = data['states']

    N = len(Osymbols)
    O = [obs_symbols[j] for j in Osymbols]

    alpha = forward(pi, A, B, O)
    beta = backward(pi, A, B, O)

    prob1 = seqprob_forward(alpha)
    prob2 = seqprob_backward(beta, pi, B, O)
    print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

    viterbi_path = viterbi(pi, A, B, O)

    print('Viterbi best path is ')
    for j in viterbi_path:
        print(states_symbols[j], end=' ')


if __name__ == "__main__":
    main()

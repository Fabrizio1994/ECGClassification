from beatclassification.NN.Preprocessing import Preprocessing
from sklearn.linear_model import LogisticRegression
prep = Preprocessing()
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from collections import defaultdict
import scikitplot as splt
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle


def mackey_glass_non_linearity(x_t, x_t_minus_tau, input):
    ni = 0.8
    gamma = 0.5
    p = 7.
    output = -x_t + (ni * (x_t_minus_tau + gamma*input)) / (1 +
                                                       math.pow(x_t_minus_tau + gamma * input, p))
    return output

def generate_ranmask(M,N):
    rand = np.random.random((M,N))
    # generate matrix of 1 , -1
    return np.vectorize(lambda x: -1 if x < 0.5 else 1)(rand)


def reservoir(dataset, dataset_name, write=True):
    if not write:
        return np.load(dataset_name + '.npy')
    N = 25
    M = 170
    #tau = 5
    prev = 0
    #state_vec = np.zeros(tau, dtype='float64')
    state_vec = defaultdict(int)
    D = list()
    for i, beat in enumerate(dataset):
        print(i)
        mask = generate_ranmask(M, N)
        S = list()
        for j, sample in enumerate(beat):
            input = sample * mask[j]
            s = list()
            for node in range(N):
                response = mackeyglass_rk4(input[node], state_vec, prev, node)
                if response == np.inf:
                    raise Exception("found:Infinity! not suitable for classifiers")
                if math.isnan(response):
                    raise Exception("found:Not a number! not suitable for classifiers")
                s.append(response)
                prev = response
                #state_vec = update_state(response, state_vec)
                state_vec[node] = response
            S.append(s)
        # print(np.mean(S))
        D.append(S)
    D = np.array(D)
    np.save(dataset_name + '.npy', D)
    #return D.reshape((len(dataset),170*25))
    return D

def update_state(response, state_vec):
    for i,e in reversed(list(enumerate(state_vec.tolist()))):
        if i > 0:
            state_vec[i] = state_vec[i-1]
        else:
            state_vec[i] = response
    return state_vec


def mackeyglass_rk4(input, state_vec, prev, node):
    theta = 0.2
    # tau = 5
    x_t = prev
    x_t_minus_tau = state_vec[node]
    k1 = theta*mackey_glass_non_linearity(x_t, x_t_minus_tau, input)
    k2 = theta*mackey_glass_non_linearity(x_t+0.5*k1, x_t_minus_tau, input)
    k3 = theta*mackey_glass_non_linearity(x_t+0.5*k2, x_t_minus_tau, input)
    k4 = theta*mackey_glass_non_linearity(x_t+k3, x_t_minus_tau, input)
    return x_t + k1/6 + k2/3 + k3/3 + k4/6


X_train, Y_train, X_test, Y_test = prep.preprocess(one_hot=False)
assert X_train.shape[0] == Y_train.shape[0]
X_train = reservoir(X_train, 'train', write=False)
shape = X_train.shape
X_train = X_train.reshape(shape[0], shape[1]*shape[2])
X_test = reservoir(X_test, 'test', write=False)
shape = X_test.shape
X_test = X_test.reshape((shape[0], shape[1]*shape[2]))
lr = LogisticRegression(penalty='l1', verbose=1, solver='saga', n_jobs=8, max_iter=100000, tol=0.000001)
lr.fit(X_train, Y_train)
pred = lr.predict(X_test)
splt.metrics.plot_confusion_matrix(Y_test, pred)
with open('logistic_regression.pkl', 'wb') as fid:
    pickle.dump(lr, fid)
print(precision_score(Y_test, pred, average=None))
print(recall_score(Y_test, pred,average=None))
plt.show()

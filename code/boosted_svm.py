import numpy as np
import cvxopt
from kernel_svm import cvxopt_solve_qp
import random
from pca import *
from mda import *

def WeakLinearSVM(X_train, y_train):
    L = X_train.shape[0]
    N = X_train.shape[2]
    K = np.zeros((N,N))
    P = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i,j] = np.matmul(X_train[:,:,i].T,X_train[:,:,j]).reshape(1,)
            P[i,j] = y_train[i]* K[i,j] * y_train[j]
    q = -1*np.ones((N,1))
    G = -1*np.eye(N)
    h = np.zeros((N,1))
    mu = cvxopt_solve_qp(P, q, G, h)
    non_zero = np.nonzero(mu)[0]
    Theta = np.zeros((L,1))
    for n in range(N):
        Theta += mu[n]*y_train[n]*X_train[:,:,n]
    f_train = 0
    for n in range(N):
        f_train += mu[n]*y_train[n]*K[non_zero[0],n]
    theta_0 = (y_train[non_zero[0]] - f_train)[0]
    return theta_0, Theta


def boosted_svm(X_train, y_train, X_test, y_test, K):
    L = X_train.shape[0]
    num_test = X_test.shape[2]
    N = X_train.shape[2]
    w = np.ones((N,1))
    P = np.zeros((N,1))
    phi = np.zeros((N,1))
    a = 0
    y_train = y_train.reshape(y_train.shape[0],1)
    F = np.zeros((num_test,1))
    for i in range(K):
        train_subset = random.sample(range(0,300), 50)
        theta_0, Theta = WeakLinearSVM(X_train[:,:,train_subset], y_train[train_subset])
        # Weak Classifier
        phi = np.sign(theta_0 + np.matmul(Theta.T, X_train.reshape(L,N)).T)
        print(phi.shape)
        print(y_train.shape)
        # Probability of misclassification
        P = w/w.sum(axis=0)
        epsilon = np.matmul(P.T, (y_train != phi))
        
        a = 0.5*np.log((1-epsilon)/epsilon)
        # Update weights

        for n in range(N):
            # print(y_train.shape , phi.shape , w.shape)
            w[n] = w[n]*np.exp(-a * y_train[n] * phi[n])
        F += a * np.sign(theta_0 + np.matmul(Theta.T, X_test.reshape(L,num_test)).T)    
        print(theta_0)
        print(Theta.shape)
    y_pred = np.sign(F)
    # print(y_pred.shape , y_test.shape)
    
    y_test = y_test.reshape(100,1)
    test_acc = np.mean(y_pred == y_test)*100
    return y_pred, test_acc

if __name__ == "__main__":
    X_train , X_test , y_train, y_test = split_2()
    # print(X_train.shape , X_test.shape , y_train.shape , y_test.shape)
    X_train_pca = pca(X_train)
    X_test_pca = pca(X_test)
    # print(y_train.reshape(y_train.shape[0], ))
    y_pred , test_acc = boosted_svm(X_train_pca , y_train, X_test_pca  , y_test ,5)
    print(test_acc)
    



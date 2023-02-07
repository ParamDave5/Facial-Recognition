import numpy as np
from pca import *
from mda import *
import cvxopt
from loadData import *

# def non_linear_optimizer(P,q,G,h,A=None,b=None):
#     P = .5 * (P + P.T)  # make sure P is symmetric
#     args = [cvxopt.matrix(P), cvxopt.matrix(q)]
#     args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
#     if A is not None:
#         args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
#     sol = cvxopt.solvers.qp(*args)
#     if 'optimal' not in sol['status']:
#         return None
#     return np.array(sol['x']).reshape((P.shape[1],1))

# def kernel_svm(X_train, y_train, X_test, y_test, kernel, sig, r):
#     L = X_train.shape[0]
#     N = X_train.shape[2]
#     K = np.zeros((N,N))
#     P = np.zeros((N,N))
#     for i in range(N):
#         for j in range(N):
#             if kernel == 1:
#                 K[i,j] = np.exp(-(1/sig**2)*np.linalg.norm(X_train[:,:,i] - X_train[:,:,j])**2)
#             elif kernel == 2:
#                 K[i,j] = pow((np.matmul(X_train[:,:,i].T,X_train[:,:,j]) + 1), r).reshape(1,)
#             P[i,j] = y_train[i]* K[i,j] * y_train[j]

#     q = -1*np.ones((N,1))
#     G = -1*np.eye(N)
#     h = np.zeros((N,1))
#     mU = non_linear_optimizer(P, q, G, h)
#     non_zero = np.nonzero(mU)[0]
#     num_test = X_test.shape[2]
#     f_test = np.zeros((num_test, 1))
#     for j in range(num_test):
#         for n in range(N):
#             if kernel == 1:
#                 K_test = np.exp(-(1/sig**2)*np.linalg.norm(X_test[:,:,j] - X_train[:,:,n])**2)
#             elif kernel == 2:
#                 K_test = pow((np.matmul(X_test[:,:,j].T,X_train[:,:,n]) + 1), r).reshape(1,)
#             f_test[j] += mU[n]*y_train[n]*K_test

#     f_train = 0
#     for n in range(N):
#         f_train += mU[n]*y_train[n]*K[non_zero[0],n]
#     theta_0 = (y_train[non_zero[0]] - f_train)[0]
#     y_pred = np.sign(theta_0*np.ones((num_test, 1)) + f_test)
#     test_acc = np.mean(y_pred == y_test)*100
#     return y_pred, test_acc


def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],1))


def kernel_svm(X_train, y_train, X_test, y_test, kernel, sig, r):
    
    L = X_train.shape[0]
    N = X_train.shape[2]
    K = np.zeros((N,N))
    P = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if kernel == 1:
                K[i,j] = np.exp(-(1/sig**2)*np.linalg.norm(X_train[:,:,i] - X_train[:,:,j])**2)
            elif kernel == 2:
                K[i,j] = pow((np.matmul(X_train[:,:,i].T,X_train[:,:,j]) + 1), r).reshape(1,)
            P[i,j] = y_train[i]* K[i,j] * y_train[j]
    q = -1*np.ones((N,1))
    G = -1*np.eye(N)
    h = np.zeros((N,1))
    mu = cvxopt_solve_qp(P, q, G, h)
    non_zero = np.nonzero(mu)[0]
    num_test = X_test.shape[2]
    f_test = np.zeros((num_test, 1))
    for j in range(num_test):
        for n in range(N):
            if kernel == 1:
                K_test = np.exp(-(1/sig**2)*np.linalg.norm(X_test[:,:,j] - X_train[:,:,n])**2)
            elif kernel == 2:
                K_test = pow((np.matmul(X_test[:,:,j].T,X_train[:,:,n]) + 1), r).reshape(1,)
            f_test[j] += mu[n]*y_train[n]*K_test
    f_train = 0
    for n in range(N):
        f_train += mu[n]*y_train[n]*K[non_zero[0],n]
    theta_0 = (y_train[non_zero[0]] - f_train)[0]
    y_pred = np.sign(theta_0*np.ones((num_test, 1)) + f_test)
    y_pred = y_pred.reshape(100, )
    print("Y test shape" , y_test.shape)
    print("Y pred shape", y_pred.shape)

    # print(y_pred)

    test_acc = np.mean(y_pred == y_test)*100
    return y_pred, test_acc

if __name__ == "__main__":
    X_train , X_test , y_train, y_test = split_2()

    # print(X_train.shape , X_test.shape , y_train.shape , y_test.shape)
    X_train = pca(X_train)
    X_test = pca(X_test)
    
    # print(y_train.reshape(y_train.shape[0], ))

    # X_train = mda(X_train,y_train,2 ,2)
    # X_test= mda(X_test,y_test ,2,2)


    # y_pred , test_acc = Kernel_SVM(X_train , y_train, X_test , y_test ,2,10,2)

    # X_train , X_test , y_train, y_test = split_2()
    # y_pred , test_acc = kernel_svm(X_train_pca , y_train, X_test_pca  , y_test ,2,20,1)

    # y_pred , test_acc = kernel_svm(X_train_mda , y_train, X_test_mda  , y_test ,2,20,1)

    print(test_acc)










import numpy as np    
from loadData import *
from pca import *
from mda import *


def reshape(mat):
    # print("shape of mat" , mat.shape)
    shape = np.array(mat).shape
    # print("shape of mat" , shape)
    mat = mat.reshape(shape[2] , shape[0],shape[1])
    # print("reshaped mat" , mat.shape)
    return mat

def Bayes_classifier(X_train, y_train , X_test ,y_test, M):
    """_summary_

    Args:
        X_train (matrix): _description_
        X_test (matrix): _description_
        y_train (list): _description_
        y_test (list): _description_
        M (_type_): number of classes

    Returns:
        _type_: _description_
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    '''
    for data.mat 
    #X_train = 504,1,400
    #y_train = 504,1,200
    #X_test = 400,1
    #y_test = 200,1
    '''

    L = X_train.shape[0]
    N = y_train.shape[0]
    N_test = y_test.shape[0]
    means = np.zeros((L,1, M))
    priors = np.zeros((M,1))
    Sigma = np.zeros((L,L, M))
    deter = np.zeros((M,1))
    Sigma_inv = np.zeros((L,L,M))

    for i in range(M):
        if M == 2 :
            class_ind = np.where(y_train==(-1 if(i == 1) else 1))[0] # change for class
        #for 2 y_train == y_train == -1 .  
        elif M > 2:
            class_ind = np.where(y_train==i+1)[0]
        Ni = len(class_ind)
        priors[i] = Ni/N
        means[:,:,i] = X_train[:,:,class_ind].mean(axis=2)
        Sigma[:,:,i] = (Ni-1)/Ni*np.cov(X_train[:,:, class_ind].reshape(L, Ni).T, rowvar=False)
        
        if np.linalg.det(Sigma[:,:,i]) < 0.00001:
            threshold = 0.0000001
            w, v = np.linalg.eig(Sigma[:,:,i])
            deter[i] = np.product(np.real(w[w>threshold]))
            Sigma[:,:,i] = Sigma[:,:,i] + 0.0001*np.eye(L)
            if i % (M//2) == 0:
                print('inside')
        else:
            if i % (M//2) == 0:
                print('outside')
            deter[i] = np.linalg.det(Sigma[:,:,i])
        Sigma_inv[:,:,i] = np.linalg.inv(Sigma[:,:,i])
        if i % (M//2) == 0:
            print(deter[i])
    y_pred = np.zeros((N_test, 1))


    for n in range(N_test):
        likelihoods = np.zeros((M,1))
        for i in range(M):

            likelihoods[i] = -np.log((2*np.pi)**(L/2)) - 0.5 * np.log(deter[i]) - 0.5 * np.matmul((X_test[:,:,n] - means[:,:,i]).T, np.matmul(Sigma_inv[:,:,i], (X_test[:,:,n] - means[:,:,i])))

        # print(likelihoods)
        if M == 2:
            print(np.argmax(likelihoods + np.log(priors)) + 1)
            y_pred[n] = (-1 if np.argmax(likelihoods + np.log(priors)) + 1 == 2 else 1) 

        elif M > 2:
            y_pred[n] = np.argmax(likelihoods + np.log(priors)) + 1  # for 2 not 
    
    y_test = y_test.reshape(100,1)
    return y_pred, np.mean(y_pred == y_test)*100


if __name__ == "__main__":
        # X_train , X_test , y_train , y_test = split(600,3)
        X_train , X_test , y_train , y_test = split_2()
        # pca_output_X_train = pca(X_train)
        # pca_output_X_test = pca(X_test)

        mda_output_X_train = mda(X_train,y_train ,2,2)
        mda_output_X_test = mda(X_test,y_test,2,2)
        # mda_output_X_train = MDA(X_train,y_train ,200 , 1)
        # mda_output_X_test = MDA(X_test,y_test,200 , 1)


        # y_pred , acc_mda = Bayes_classifier(pca_output_X_train ,y_train , pca_output_X_test  , y_test ,2)
        # y_pred_mda , acc_mda = Bayes_classifier(pca_output_X_train ,y_train , pca_output_X_test  , y_test ,2)
        y_pred_mda , acc_mda = Bayes_classifier(mda_output_X_train ,y_train , mda_output_X_test  , y_test ,2)

        # print(acc)
        print(acc_mda)


import numpy as np
import matplotlib.pyplot as plt
from loadData import * 

def mda(data,y,M,classification):
    # mean of each class
    """_summary_

    Args:
        data (Array): X_train or X_test 
        y (Array): y_train , y_test
        M (int): number of class
        classification (int): 1 or 2

    Returns:
        _type_: _description_
    """
    if classification == 1 :
        L = data.shape[0] # No. of features
        N = y.shape[0] # No. of classes
        means = np.zeros((L,1, M))
        priors = np.zeros((M,1))
        anchor_mean = np.zeros((L,1))
        Sigma_i = np.zeros((L,L, M))
        
        for i in range(M):
            Ni = np.count_nonzero(y==i+1)
            class_ind = np.where(y==i+1)[0]
            priors[i] = Ni/N
            
            means[:,:,i]  = (1/Ni)*data[:, :, class_ind].sum(axis=2).reshape(L,1)
            anchor_mean += priors[i]*means[:,:,i]
            Sigma_i[:,:,i] = (Ni-1)/Ni * np.cov(data[:,:,class_ind].reshape(data[:,:,class_ind].shape[0], data[:,:,class_ind].shape[2]).T, rowvar = False) 
        
        Sigma_b = np.zeros((L,L))
        Sigma_w = np.zeros((L,L))
        for i in range(M):
            Sigma_b += priors[i] * np.matmul(means[:,:,i] - anchor_mean, (means[:,:,i] - anchor_mean).T)
            Sigma_w += priors[i] * Sigma_i[:,:,i]
        if np.linalg.det(Sigma_w) == 0:
            Sigma_w += 0.0001*np.eye(L)
        
        # ------------- Top m Eigenvectors of Sigma_w^(-1) Sigma_b ------------
        W, V = np.linalg.eig(np.matmul(np.linalg.inv(Sigma_w), Sigma_b))
        m = np.count_nonzero(np.real(W) > 1e-10) # m <= M-1, where, M is no. of classes
        idx = np.argsort(np.real(W))[::-1]
        sorted_V = V[:,idx]
        A = sorted_V[:,:m]
        Theta = (1/L)*A

        Z = np.matmul(Theta.T,  data.reshape(data.shape[0], data.shape[2])).reshape(m,N)
        X = np.matmul(Theta, Z)
        X = X.reshape(X.shape[0], 1, X.shape[1])

        return np.real(X)

    if classification == 2 :

        num = data.shape[2]
        print(data.shape)
        data = data.reshape(504,1,num)
        data_ = data.reshape(504,num)
        cluster_means = []
        sigma_I = []
        cluster_means = np.zeros((504,1,2))
        j=0
        y = y.reshape(y.shape[0],)
        # print(y)
        for i in range(-1,2,2):
            
            Ni = np.count_nonzero(y==i)
            class_idx = np.where(y==i)
            class_idx = class_idx[0] 
            # Ni = len(class_idx)
            cluster_means[:,:,j] = data[:,:,class_idx].mean(axis=2)    
            sigma_i = (Ni-1/Ni)*np.cov(data[:,:,class_idx].reshape(data[:,:,class_idx].shape[0],data[:,:,class_idx].shape[2]).T, rowvar = False)
            sigma_I.append(sigma_i)
        cluster_means = cluster_means.reshape(2,504,1)

        prior = 200/400
    
        anchor_mean = np.sum(cluster_means,axis = 0)*prior

        sigma_b = np.zeros((504,504))
        sigma_w = np.zeros((504,504))
        for i in range(np.array(cluster_means).shape[0]):

            sigma_b += np.dot((cluster_means[i]- anchor_mean),(cluster_means[i]- anchor_mean).T)*prior
            sigma_w += sigma_I[i]*prior

        sigma_w += 0.000001 * np.eye(504)   
        
        _ ,theta   = np.linalg.eig(np.dot(np.linalg.inv(sigma_w),sigma_b))
        theta = np.real(theta)*(1/504)
        final_out = theta.T.dot(data_)
        reproj = theta.dot(final_out)
        reproj = reproj.reshape(504,1,num)
        return reproj

if __name__ == "__main__":
    mat = read_data()
    X_train , X_test , y_train , y_test = split(600,3)
    mda_ = mda(X_train,X_test , 1)

    # print(np.array(mda_).shape)
    # mda_ = mda_.reshape(24,21,400)
    plt.imshow(mda_[:,:,1].reshape(24,21),cmap = 'gray')
    plt.savefig()
    plt.show()

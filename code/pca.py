import numpy as np
from loadData import * 
import matplotlib.pyplot as plt 

def pca(data):
    """_summary_

    Args:
        data (matrix): matrix of data on which pca is to be done

    Returns:
        _type_: reconstructed data matrix (504,1, )
    """

    num = data.shape[2]
    # print("num", num)
    
    #reshape
    print(data.shape)

    x = data.reshape(504,1,num)

    # print("shape of x, ",x.shape)
    #mean
    mean =  np.mean(x , axis = 2)
    mean = mean.reshape(504,1)
    mean_ = mean[:,:, None]
    # print("Mean shape" , mean.shape)
    #centering
    centered = np. subtract(x,mean_) 
    print("shape of centered", centered.shape)
    #covar
    centered_copy = centered.reshape(504,num)

    covar = np.cov(centered_copy.T, rowvar=False)
    # print("shape of cov" , covar.shape)
    #eigvalues 
    eigval , eigvec = np.linalg.eig(covar) 
    idx  = np.argsort(eigval)[::-1] #sort desc
    eigvec = eigvec[:,idx]
    eigvec = eigvec[:,:100] #take first 100, 100 is our hyperparameter. 

    # eigvect.centered 
    # print("shape of eigenvee" ,eigvec.shape)

    eigen_faces = eigvec.T.dot(centered_copy)
    # print("eigen_face" , eigen_faces.shape)

    eigen_faces = np.matmul(eigvec , eigen_faces)
    # print("eigen_face" , eigen_faces.shape)

    #add mean
    final_output = eigen_faces + mean
    # print("final" , final_output.shape)

    # take real values 
    final_output = np.real(final_output)
    # print("Shape of final",final_output.shape)
    final_output = final_output.reshape(504,1,num)
    return final_output

if __name__ == "__main__":    
    mat = read_data()

    X_train , X_test , y_train , y_test = split(600,3)

    pca_output = pca(X_train)

    plt.imshow(pca_output[:,:,4].reshape(24,21),cmap='gray')
    plt.savefig("pca.png")
    plt.show()







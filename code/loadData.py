import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random 
# mention which data is to be loaded the data loader
def read_data():
    """_summary_

    Returns:
        _type_: return data.mat with dimensions 600,24,21
    """
    mat = sio.loadmat('/Users/sheriarty/Desktop/sprProj1/data.mat')
    mat = np.array(mat["face"])
    # plt.imshow(mat[:,:,4].reshape(24,21),cmap='gray')
    # plt.savefig("og.png")
    # plt.show()

    return mat

def read_expression():
    """_summary_

    Returns:
        _type_: return illimination.mat data
    """
    mat = sio.loadmat('/Users/sheriarty/Desktop/sprProj1/illumination.mat')
    mat = np.array(mat["illum"])

    return mat

def read_pose():
    """_summary_

    Returns:
        _type_: return pose.mat data
    """
    mat = sio.loadmat('/Users/sheriarty/Desktop/sprProj1/pose.mat')
    mat =  np.array(mat["pose"])
    return mat

def return_label():
    count = 1
    labels = []
    for i in range(1,201):
        for j in range(0,3):
            labels.append(i)
    return labels

#dataloader for data.mat
def split(total_data , n):
    """_summary_
    Inputs: 
        total_data : total data points
        n : number of data points in each class
         
    Returns:
        _type_: returns 
                X_train: a list of all train indices in data.mat (504,1,400)
                X_test : a list of all test indices in data.mat (504,1,200)
                y_train : a list of labels corresponding to train list (400,1)
                y_test : a list of labels corresponding to test list (200,1)
    """
    # 1 to 600
    data = read_data()
    N = data.shape[2]
    train_idx = []
    test_idx = []
    ips = 3
    M = 200 
    data = data.reshape(data.shape[0]*data.shape[1],1,data.shape[2])
    labels = np.zeros((data.shape[2],1))

    for i in range(M):
        labels[3*(i+1)-3] = i+1
        labels[3*(i+1)-2] = i+1
        labels[3*(i+1)-1] = i+1

    for i in range(M):
        rand = random.sample(range(0,ips),int(2/3*ips))
        for j in range(int(2/3*(ips))):
            train_idx.append(i*ips + rand[j])
    
    for i in range(N):
        if i not in train_idx:
            test_idx.append(i)

    X_train = data[:,:,train_idx].reshape(504,1, 400)
    y_train = labels[train_idx].reshape(400 , 1)
    X_test = data[:,:,test_idx].reshape(504,1,200)
    y_test = labels[test_idx].reshape(200, 1)
    
    return X_train , X_test , y_train , y_test 


def split_2():
    #load data.mat
    data = read_data()
    M = 200
    data.reshape(data.shape[0]*data.shape[1],1,data.shape[2])
    #smile is second image. i.e 1,4,7 N*3+1
    new_data = np.zeros((data.shape[0],data.shape[1],2*data.shape[2]//3))
    label = np.zeros((new_data.shape[2],1))

    j = 0
    for i in range(data.shape[2]):
        if (i+1)%3 != 0 :
            new_data[:,:,j] = data[:,:,i]
            if (i+1)%3 == 1 :
                label[j] = -1
            elif (i+1)%3 == 2:
                label[j] = 1
            j +=1

    # label = np.array(label.reshape((400,)))
    # new_data = np.array(new_data)
    # X_train , X_test , y_train , y_test = new_data[:,:,:300].reshape(504,1,300 ) , new_data[:,:,300:].reshape(504,1, 100) , label[:300] , label[300:]
    X_train = new_data[:,:,:3*(new_data.shape[2])//4].reshape(504,1,300)
    y_train = label[:3*(new_data.shape[2])//4].reshape(300, )
    X_test = new_data[:,:,3*(new_data.shape[2])//4 :].reshape(504,1,100)
    y_test = label[3*(new_data.shape[2])//4 :].reshape(100, )
    
    return X_train , X_test , y_train , y_test 


    

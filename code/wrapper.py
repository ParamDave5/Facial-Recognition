import numpy
from loadData import *
from pca import *
from mda import *
from bayesian import *
from knn import *
from kernel_svm import *
from boosted_svm import *

#choose what classification to be done
#choose compression method
#choose classifier 

hyperparameter = False

classification = int(input('Enter type of classification (1: Face Recognition, 2: Neutral vs Expression): '))
compression = str(input('Enter compression method ("PCA" , "MDA")'))
classifier = str(input("Choose a classifier (Bayes , KNN , Kernel_SVM, Boosted_SVM)"))
if classifier == 'KNN':
    K = int(input("enter the nearest neighbour in KNN"))
if classifier == 'Kernel_SVM':
    kernel = int(input("enter the kernel type (1: Radial Bias , 2: Polynomial)"))


if classification == 1:
    X_train , X_test , y_train , y_test = split(600,3) 
    classes = 200
    
elif classification ==2 :
    X_train , X_test , y_train , y_test  = split_2()
    classes = 2

if compression == "PCA":
    X_train = pca(X_train)
    X_test = pca(X_test)
elif compression == "MDA":
    X_train = mda(X_train, y_train ,classes, classification)
    X_test = mda(X_test ,y_test,classes, classification)

if classifier == 'Bayes': 
    print("--- Performing Bayes Classification ----")
    y_pred , accuracy = Bayes_classifier(X_train ,y_train , X_test  , y_test ,classes)
    print("----Test Accuracy----")
    print(accuracy)

if classifier == 'KNN':

    print("----Performing KNN----")
    knn = KNearestNeighbor()
    knn.train(X_train, y_train)
    y_pred = knn.predict(X_test, K)
    y_test = y_test.reshape(y_test.shape[0],)
    test_accuracy = np.mean(y_pred == y_test)*100
    print('----Test accuracy----')
    print(test_accuracy)
    test_acc = []

    if hyperparameter == True :
        
        k = [1,2,3,4,5]
        for i in range(len(k)):
            print("performing KNN Hyperparameter")
            knn = KNearestNeighbor()
            knn.train(X_train, y_train)
            y_pred = knn.predict(X_test, k[i])
            y_test = y_test.reshape(y_test.shape[0],)
            test_accuracy = np.mean(y_pred == y_test)*100
            test_acc.append(test_accuracy)
    print(test_acc)
    print("Best accuracy is: ", max(test_acc))



if classifier == 'Kernel_SVM':
    print("----Performing Kernel SVM----")
    sigma = 20
    r = 2
    y_pred , test_accuracy = kernel_svm(X_train, y_train, X_test, y_test, kernel, sigma, r)
    print('----Test accuracy----')
    print(test_accuracy)
    if hyperparameter == True :
        sigma = [5,10,15,20,25] 
        r = [1,2,3,4,5]
        test_acc = []
        for i in range(len(sigma)):
            y_pred , test_accuracy = kernel_svm(X_train, y_train, X_test, y_test, kernel, sigma[i], r[i])
            test_acc.append(test_accuracy)
        print(test_acc)
        print("Best Kernel_SVM Accuracy is: ", max(test_acc))



if classifier == 'Boosted_SVM':
    print("----PErforming Booster SVM----")
    k = 5
    y_pred , test_accuracy = boosted_svm(X_train , y_train , X_test , y_test , k)
    print(test_accuracy)
    if hyperparameter == True :
        k = [5,10,15,20]
        test_acc = []
        for i in range(len(k)):
            y_pred , test_accuracy = boosted_svm(X_train, y_train, X_test, y_test, k[i])
            test_acc.append(test_accuracy)
        print(test_acc)
        print("Best Kernel_SVM Accuracy is: ", max(test_acc))
    



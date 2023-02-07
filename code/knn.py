import numpy as np    
from loadData import *
from pca import *
from mda import *

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (D, 1, num_test) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X.reshape(X.shape[0], X.shape[2]).T
        self.y_train = y.reshape(y.shape[0],)

    def predict(self, X, k=1):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (D, 1, num_test) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        X = X.reshape(X.shape[0], X.shape[2]).T
        dists = self.compute_distances_no_loops(X)
        
        return self.predict_labels(dists, k=k)


    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Take sum of squares of all elements of test data
        sx = np.sum(X**2, axis=1, keepdims=True)
        # Take sum of squares of all elements of training data
        sx_train = np.sum(self.X_train**2, axis=1, keepdims=True)
        # Subtract 2*X*X_train.T to get -2*x[i]*x_train[j] kind of terms
        # Now using the formula (a-b)^2 =  a^2 + b^2 - 2*a*b
        # And taking square root we get L2 norm 
        dists = np.sqrt(sx + sx_train.T - 2*X.dot(self.X_train.T))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # Sort indices of test data according to increassing order of distances 
            dist_idx = np.argsort(dists[i,:])
            # Take the training y_train corresponding to 
            # first k (nearest k) elements/neighbours from above index list
            closest_y  = list(self.y_train[dist_idx[:k]])
            

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # The y that occurs the most no. of times in closest_y (Mode of closest_y) 
            # Gives the prediction y_pred[i]
            y_pred[i] = max(set(closest_y), key = closest_y.count)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred



if __name__ == "__main__":
    # X_train , X_test , y_train , y_test = split(600,3)
    X_train , X_test , y_train , y_test = split_2()

    pca_output_X_train = pca(X_train)
    pca_output_X_test = pca(X_test)

    mda_output_X_train = mda(X_train,y_train,2)
    mda_output_X_test = mda(X_test,y_test,2)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    knn = KNearestNeighbor()

    # knn.train(pca_output_X_train, y_train)
    # y_pred = knn.predict(pca_output_X_test, 1)

    knn.train(mda_output_X_train, y_train)
    y_pred = knn.predict(mda_output_X_test, 1) 

    y_test = y_test.reshape(np.array(y_test).shape[0],)
    test_acc = np.mean(y_pred == y_test)*100
    print('--------Test accuracy----------')
    print(test_acc)
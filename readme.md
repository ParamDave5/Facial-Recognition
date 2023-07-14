
### This is a repositort for face recognition and facial expression detection using Bayes Classifier , KNN , Kernel SVM and Boosted SVM using PCA and MDA as preprocessing methods

Data should be in a parent folder of code.  

Enter path to data.mat in line 12 of function read_data in loadData.py

RUN "python wrapper.py"

Enter the inputs in terminal 
    1)Classifier 1 or 2 i.e Face Detection or Facial Expression Detection
    2)Compression MEthod PCA or MDA
    3)Classifier: Bayes, KNN , Kernel_SVM, Boosted_SVM
    4)Enter HyperParameters according to the Classifier

For HyperParameter tuning code set hyperparameter flag = True

Libraries Needed: cvxopt,scipy , python
Installation Guide for cvxopt

"pip install cvxopt"

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    train_data = np.concatenate((np.ones(n_data).reshape(n_data, 1), train_data), axis=1)

    pred = np.reshape(sigmoid(np.dot(train_data, initialWeights)), (-1, 1))
    error = -1 * (1 / n_data) * np.sum(((labeli * np.log(pred)) + ((1 - labeli) * np.log(1 - pred))))
    error_grad = (1 / n_data) * (np.dot(train_data.T, np.subtract(pred, labeli)))
    
    return error, error_grad.ravel()


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    n_data = data.shape[0]

    # Appending bias to the data
    data = np.concatenate((np.ones(n_data).reshape(n_data, 1), data), axis=1)

    result = np.zeros((data.shape[0], 1))
    label = sigmoid(np.dot(data, W))

    label = np.argmax(label, axis=1)
    label = np.reshape(label, (n_data, 1))

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    iw = np.reshape(params, (n_feature+1, 10))

    # Appending bias to the data
    train_data = np.concatenate((np.ones(n_data).reshape(n_data, 1), train_data), axis=1)

    theta = np.exp(np.dot(train_data, iw)) / np.reshape(np.sum(np.exp(np.dot(train_data, iw)), axis=1), (n_data, 1))

    sum_inner = np.sum(labeli * np.log(theta))
    error = -1 * np.sum(sum_inner)

    error_grad = np.dot(train_data.T, (theta - labeli))

    return error, error_grad.ravel()


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    n_data = data.shape[0]

    # Appending bias to the data
    data = np.concatenate((np.ones(n_data).reshape(n_data, 1), data), axis=1)

    theta = np.exp(np.dot(data, W)) / np.sum(np.exp(np.dot(data, W)))
    label = np.argmax(theta, axis=1)
    label = np.reshape(label, (n_data, 1))

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))



def errorCountChecker(predLabel,realLabel):
    error = np.reshape(np.zeros(10), (10, 1))
    count = np.reshape(np.zeros(10), (10, 1))
    n_data = len(realLabel)
    for i in range(n_data) :
        count[int(realLabel[i])][0] += 1
        predL = predLabel[i][0]
        realL = realLabel[i][0]
#        print("predL: ",predL,"realL: ", realL)
        if predL != realL:
            error[int(realLabel[i])][0] = error[int(realLabel[i])][0] + 1
            
            
    return error,count
    
    
# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

errorForTrain,countForTrain = errorCountChecker(predicted_label,train_label)

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

errorForValidation,countForValidation = errorCountChecker(predicted_label,validation_label)

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


errorForTest,countForTest = errorCountChecker(predicted_label,test_label)








"""
Script for Support Vector Machine
"""


print('\n\n--------------SVM-------------------\n\n')

##################
# YOUR CODE HERE #
##################

# Randomly selecting the 10,000 samples.
indx = np.random.randint(50000, size=10000)
train_svmdata = train_data[indx, :]
train_svmlab = train_label[indx, :]

# ----------- Linear Kernel ----------------

print('\n Linear Kernel : \n')

model_lin = SVC(kernel='linear')
model_lin.fit(train_svmdata, train_svmlab.ravel())


print('\n Training Accuracy:' + str(100 * model_lin.score(train_svmdata, train_svmlab)) + '%')
print('\n Validation Accuracy:' + str(100 * model_lin.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy:' + str(100 * model_lin.score(test_data, test_label)) + '%')

# ------------ Radial Basis Function with gamma=1.0 -----------------
print('\n Radial Basis Function Kernel gamma= 1 : \n')

model_rbf = SVC(kernel='rbf', gamma=1.0)
model_rbf.fit(train_svmdata, train_svmlab.ravel())

print('\n Training Accuracy:' + str(100 * model_rbf.score(train_svmdata, train_svmlab)) + '%')
print('\n Validation Accuracy:' + str(100 * model_rbf.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy:' + str(100 * model_rbf.score(test_data, test_label)) + '%')


# ----------- Radial Basis Function with gamma = auto---------------
print('\n Radial Basis Function Kernel gamma = "auto": \n')

model_rbf2 = SVC(kernel='rbf', gamma='auto')
model_rbf2.fit(train_svmdata, train_svmlab.ravel())

print('\n Training Accuracy:' + str(100 * model_rbf2.score(train_svmdata, train_svmlab)) + '%')
print('\n Validation Accuracy:' + str(100 * model_rbf2.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy:' + str(100 * model_rbf2.score(test_data, test_label)) + '%')

# ---------------- Radial Basis Function with gamma = default and changing C parameter ------------------
accura = np.zeros((11, 3), float)
C_vals = np.array([1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
inp = 0

# Loop for changing C values
for i in C_vals:
    print("Value of C is: \n", i)
    model_rbfC = SVC(kernel='rbf', C=i)
    model_rbfC.fit(train_svmdata, train_svmlab.ravel())
    if inp < 11:
        accura[inp][0] = 100 * model_rbfC.score(train_svmdata, train_svmlab)
        accura[inp][1] = 100 * model_rbfC.score(validation_data, validation_label)
        accura[inp][2] = 100 * model_rbfC.score(test_data, test_label)
        print('\n Radial Basis Function Kernel for C = ' + str(i) + ' \n')
        print('\n Training Accuracy:' + str(accura[inp][0]) + '%')
        print('\n Validation Accuracy:' + str(accura[inp][1]) + '%')
        print('\n Testing Accuracy:' + str(accura[inp][2]) + '%')

    inp = inp + 1

plt.plot(C_vals, accura)

plt.legend(['training_data', 'validation_data', 'test_data'])

## Complete training set application:

model_rbf_whole = SVC(kernel='rbf', gamma='auto', C=40)
model_rbf_whole.fit(train_data, train_label.ravel())

print('\n Radial Basis Function with whole training set with best parameters : \n')
print('\n Training Accuracy:' + str(100 * model_rbf_whole.score(train_data, train_label)) + '%')
print('\n Validation Accuracy:' + str(100 * model_rbf_whole.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy:' + str(100 * model_rbf_whole.score(test_data, test_label)) + '%')




## Complete training set application: gamma= 'scale'

model_rbf_whole_scale = SVC(kernel='rbf', gamma='scale', C=40)
model_rbf_whole_scale.fit(train_data, train_label.ravel())

print('\n Radial Basis Function with whole training set with best parameters : \n')
print('\n Training Accuracy:' + str(100 * model_rbf_whole_scale.score(train_data, train_label)) + '%')
print('\n Validation Accuracy:' + str(100 * model_rbf_whole_scale.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy:' + str(100 * model_rbf_whole_scale.score(test_data, test_label)) + '%')

##################
##################

"""
Script for Extra Credit Part
"""

#################
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
errorForTrainMLR,countForTrainMLR = errorCountChecker(predicted_label_b,train_label)




# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')
errorForValidationMLR,countForValidationMLR = errorCountChecker(predicted_label_b,validation_label)


# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
errorForTestMLR,countForTestMLR = errorCountChecker(predicted_label_b,test_label)


#################

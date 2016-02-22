import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  #your code here
    
    

def preprocess():
    """ Input:
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

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary

    """Stacks all training matrices into one 60000 x 784 matrix and normalizes it to the range [0,1]"""
    allTrain = np.concatenate((mat.get('train0'),mat.get('train1'),mat.get('train2'),mat.get('train3'),mat.get('train4'),mat.get('train5'),mat.get('train6'),mat.get('train7'),mat.get('train8'),mat.get('train9')), axis=0)
    allTrain = allTrain/255.0
    
    """Creates a 60000 x 10 label matrix corresponding to the allTrain matrix.  They are essentially in parallel so for example, the data at index 5 in allTrain will match up with the label at index 5 in allTrainLabel.  Each row is 10 units long and corresponds to a digit 0-9."""
    allTrainLabel = np.concatenate(((np.full((len(mat.get('train0')), 10), [1,0,0,0,0,0,0,0,0,0])),(np.full((len(mat.get('train1')), 10), [0,1,0,0,0,0,0,0,0,0])),(np.full((len(mat.get('train2')), 10), [0,0,1,0,0,0,0,0,0,0])),(np.full((len(mat.get('train3')), 10), [0,0,0,1,0,0,0,0,0,0])),(np.full((len(mat.get('train4')), 10), [0,0,0,0,1,0,0,0,0,0])),(np.full((len(mat.get('train5')), 10), [0,0,0,0,0,1,0,0,0,0])),(np.full((len(mat.get('train6')), 10), [0,0,0,0,0,0,1,0,0,0])),(np.full((len(mat.get('train7')), 10), [0,0,0,0,0,0,0,1,0,0])),(np.full((len(mat.get('train8')), 10), [0,0,0,0,0,0,0,0,1,0])),(np.full((len(mat.get('train9')), 10), [0,0,0,0,0,0,0,0,0,1]))), axis=0)

    """This splits and shuffles both the data and label matrixes in the same order so they will still match up into a validation set and a training set"""
    seed = np.random.permutation(60000)
    train_data = allTrain[seed[:50000],:]
    validation_data = allTrain[seed[50000:],:]
    train_label = allTrainLabel[seed[:50000],:]
    validation_label = allTrainLabel[seed[50000:],:]

    """Stacks all test matrices into one 10000 x 784 matrix and normalizes it to the range [0,1]"""
    test_data = np.concatenate((mat.get('test0'),mat.get('test1'),mat.get('test2'),mat.get('test3'),mat.get('test4'),mat.get('test5'),mat.get('test6'),mat.get('test7'),mat.get('test8'),mat.get('test9')), axis=0)
    test_data = test_data/255.0
    
        """Creates a 10000 x 10 label matrix corresponding to the test_data matrix.  They are essentially in parallel so for example, the data at index 5 in test_data will match up with the label at index 5 in test_label.  Each row is 10 units long and corresponds to a digit 0-9."""
    test_label = np.concatenate(((np.full((len(mat.get('test0')), 10), [1,0,0,0,0,0,0,0,0,0])),(np.full((len(mat.get('test1')), 10), [0,1,0,0,0,0,0,0,0,0])),(np.full((len(mat.get('test2')), 10), [0,0,1,0,0,0,0,0,0,0])),(np.full((len(mat.get('test3')), 10), [0,0,0,1,0,0,0,0,0,0])),(np.full((len(mat.get('test4')), 10), [0,0,0,0,1,0,0,0,0,0])),(np.full((len(mat.get('test5')), 10), [0,0,0,0,0,1,0,0,0,0])),(np.full((len(mat.get('test6')), 10), [0,0,0,0,0,0,1,0,0,0])),(np.full((len(mat.get('test7')), 10), [0,0,0,0,0,0,0,1,0,0])),(np.full((len(mat.get('test8')), 10), [0,0,0,0,0,0,0,0,1,0])),(np.full((len(mat.get('test9')), 10), [0,0,0,0,0,0,0,0,0,1]))), axis=0)

    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    #Your code here
    #
    #
    #
    #
    #
    
    
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

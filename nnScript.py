import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from math import exp
np.set_printoptions(threshold=np.inf)

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
    

def simpleSigmoid(z):
    return 1/(1+exp(-z));

vSigmoid = np.vectorize(simpleSigmoid);    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  vSigmoid(z);
    
    
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
    allTrainLabel = np.concatenate(((np.full((len(mat.get('train0')), 1), 0)),(np.full((len(mat.get('train1')), 1), 1)),(np.full((len(mat.get('train2')), 1), 2)),(np.full((len(mat.get('train3')), 1), 3)),(np.full((len(mat.get('train4')), 1), 4)),(np.full((len(mat.get('train5')), 1), 5)),(np.full((len(mat.get('train6')), 1), 6)),(np.full((len(mat.get('train7')), 1), 7)),(np.full((len(mat.get('train8')), 1), 8)),(np.full((len(mat.get('train9')), 1), 9))), axis=0)

    full=allTrain.shape[0]
    split=allTrain.shape[0]*5/6
    """This splits and shuffles both the data and label matrixes in the same order so they will still match up into a validation set and a training set"""
    seed = np.random.permutation(full)
    train_data = allTrain[seed[:split],:]
    validation_data = allTrain[seed[split:],:]
    train_label = allTrainLabel[seed[:split],:]
    validation_label = allTrainLabel[seed[split:],:]

    """Stacks all test matrices into one 10000 x 784 matrix and normalizes it to the range [0,1]"""
    test_data = np.concatenate((mat.get('test0'),mat.get('test1'),mat.get('test2'),mat.get('test3'),mat.get('test4'),mat.get('test5'),mat.get('test6'),mat.get('test7'),mat.get('test8'),mat.get('test9')), axis=0)
    test_data = test_data/255.0
    
    find_dup= np.transpose(np.concatenate((train_data,validation_data,test_data),axis=0))

    """This segment finds all elements of the data which never change so they can be removed"""
    remove=np.ones((test_data.shape[1],1))
    count=0
    same=0
    for x in find_dup:
        oldp= x[0]
        for p in x:
            if p!=oldp:
                remove[count][0]=0
                same+=1
                break
            oldp=p
        count+=1

    r=np.zeros(test_data.shape[1]-same)
    count=0
    index=0
    for x in remove:
        if x==1:
            r[index]=count
            index+=1
        count+=1
    
    

    """Creates a 10000 x 10 label matrix corresponding to the test_data matrix.  They are essentially in parallel so for example, the data at index 5 in test_data will match up with the label at index 5 in test_label.  Each row is 10 units long and corresponds to a digit 0-9."""
    test_label = np.concatenate(((np.full((len(mat.get('test0')), 1), 0)),(np.full((len(mat.get('test1')), 1), 1)),(np.full((len(mat.get('test2')), 1), 2)),(np.full((len(mat.get('test3')), 1), 3)),(np.full((len(mat.get('test4')), 1), 4)),(np.full((len(mat.get('test5')), 1), 5)),(np.full((len(mat.get('test6')), 1), 6)),(np.full((len(mat.get('test7')), 1), 7)),(np.full((len(mat.get('test8')), 1), 8)),(np.full((len(mat.get('test9')), 1), 9))), axis=0)

    """columns that don't change between examples are removed"""
    train_data=np.delete(train_data, r, axis=1)
    validation_data=np.delete(validation_data, r, axis=1)
    test_data=np.delete(test_data, r, axis=1)
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
    
    """translates label vector of digits 0-9 into 1-K form"""
    count=0
    label10=np.zeros((training_label.shape[0],10))
    for x in training_label:
        if(x==0):
            label10[count]=[1,0,0,0,0,0,0,0,0,0]
        elif(x==1):
            label10[count]=[0,1,0,0,0,0,0,0,0,0]
        elif(x==2):
            label10[count]=[0,0,1,0,0,0,0,0,0,0]
        elif(x==3):
            label10[count]=[0,0,0,1,0,0,0,0,0,0]
        elif(x==4):
            label10[count]=[0,0,0,0,1,0,0,0,0,0]
        elif(x==5):
            label10[count]=[0,0,0,0,0,1,0,0,0,0]
        elif(x==6):
            label10[count]=[0,0,0,0,0,0,1,0,0,0]
        elif(x==7):
            label10[count]=[0,0,0,0,0,0,0,1,0,0]
        elif(x==8):
            label10[count]=[0,0,0,0,0,0,0,0,1,0]
        else:
            label10[count]=[0,0,0,0,0,0,0,0,0,1]
        count+=1
    
    w1 = params[:n_hidden * (n_input+1)].reshape((n_hidden, (n_input+1)))
    w2 = params[(n_hidden * (n_input+1)):].reshape((n_class, (n_hidden+1)))
    obj_val = 0  
    
    print('in nnobj')

    #Get bias dimension
    bias_dimension = training_data.shape[0]

    #Fill it all with ones
    bias = np.ones((bias_dimension,1))

    #Add bias to weights 
    training_data_with_bias = np.concatenate((training_data,bias),1)

    #Feed Foward Start By Multiplying Training data by weights of w1
    z2 = np.dot(training_data_with_bias,np.transpose(w1))

    #Apply Sigmoid function
    a2= sigmoid(z2)
    #Apply Another Bias Dimension to the new matrix

    #bias_dimension = a2.shape[0]
    bias = np.ones((bias_dimension,1))
    a2_bias= np.concatenate((a2,bias),1)

    #Multiply new matrix by the weights of w2
    z3 = np.dot(a2_bias,np.transpose(w2))
    
    #Apply Sigmoid Function to the new data
    y= sigmoid(z3)

    #yl-ol (element of equation (9))
    dif= label10-y
    
    #1-ol (element of equation (9))
    dif2= 1-y

    # Finish Forward Propagation
    
    #equation (15)
    obj_val = ((lambdaval/(2*y.shape[0]))*(np.sum(np.square(w1))+np.sum(np.square(w2))))+(np.sum(.5*np.sum(np.square(dif),axis=1))/y.shape[0])
    
    #column vector, equation (9)
    elem1=np.transpose(np.array(-1*dif*dif2*y,ndmin=2))

    #w2 matrix with bias cut out
    w2trim= np.delete(w2,w2.shape[1]-1,1)

    #equation (12) without multiplying the xi term yet
    elem2=(-1*(1-a2)*(a2))*(np.dot((dif*dif2*y),w2trim))

#summing up the inner part of equation (17)
    total=np.zeros_like(w1)
    for x in range(0,y.shape[0]):
        total+=np.dot(np.transpose(np.array(elem2[x],ndmin=2)),np.array(training_data_with_bias[x],ndmin=2))

    #equation (17)
    grad_w1 = (total+(lambdaval*w1))/y.shape[0]

    #equation (16)
    grad_w2 = (np.dot(elem1,a2_bias)+(lambdaval*w2))/y.shape[0]

    
    
        
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    print (obj_val)
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
    #Get bias dimension
    bias_dimension = data.shape[0]

    #Fill it all with ones
    bias = np.ones((bias_dimension,1))

    #Add bias to weights 
    data_with_bias = np.concatenate((data,bias),1)

    #Feed Foward Start By Multiplying Training data by weights of w1
    z2 = np.dot(data_with_bias,np.transpose(w1))

    #Apply Sigmoid function
    a2= sigmoid(z2)
    #Apply Another Bias Dimension to the new matrix

    #bias_dimension=a2.shape[0]
    #bias = np.ones((bias_dimension,1))
    a2_bias= np.concatenate((a2,bias),1)

    #Multiply new matrix by the weights of w2
    z3 = np.dot(a2_bias,np.transpose(w2))
    
    #Apply Sigmoid Function to the new data
    y= sigmoid(z3)

    #find max value and add that digit to the labels vector
    labels= np.zeros((y.shape[0],1))
    count=0
    for x in y:
        index=0
        max=0
        inmax=0
        for p in x:
            if p >= max:
                max=p
                inmax=index
            index+=1
        labels[count][0]=inmax
        count+=1
    
    print('results n ', labels)
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 16;
                   
# set the number of nodes in output unit
n_class = 10;                   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = .3;

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

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

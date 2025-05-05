# deep-learning-visual-computing-assignment-2-multi-layer-neural-networks-solved
**TO GET THIS SOLUTION VISIT:** [Deep Learning Visual Computing Assignment 2-Multi-layer neural networks Solved](https://www.ankitcodinghub.com/product/deep-learning-visual-computing-assignment-2-multi-layer-neural-networks-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;99775&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Deep Learning Visual Computing Assignment 2-Multi-layer neural networks Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
<div class="page" title="Page 1">
<div class="section">
<div class="section">
<div class="layoutArea">
<div class="column"></div>
</div>
<div class="layoutArea">
<div class="column">
Assignment 2

In this assignment we will be build a multi layer neural network and train it to classify hand-written digits into 10 classes (digits 0-9). We will use the MNIST dataset of handwritten digits for training the classifier. The dataset is a good example of real-world data and is popular in the Machine Learning community.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [186]:

</div>
</div>
<div class="layoutArea">
<div class="column">
Load and Visualize Data

MNIST dataset contains grayscale samples of handwritten digits of size 28 examples.

<pre>trX.shape:  (60000, 1, 28, 28)
trY.shape:  (60000,)
tsX.shape:  (10000, 1, 28, 28)
tsY.shape:  (10000,)
</pre>
<pre>trX.shape:  (784, 2000)
trY.shape:  (1, 2000)
tsX.shape:  (784, 2000)
tsY.shape:  (1, 2000)
Train max: value = 1.0, Train min: value = -1.0
Test max: value = 1.0, Test min: value = -1.0
Unique labels in train:  [0 1 2 3 4 5 6 7 8 9]
Unique labels in test:  [0 1 2 3 4 5 6 7 8 9]
</pre>
<pre>Displaying a few samples
labels
[[0 9 0 5 0 7 0 0 5 6]
</pre>
<pre> [0 5 4 4 6 7 3 0 9 7]
 [8 8 8 7 6 2 2 1 9 6]
 [8 1 7 2 0 3 5 2 7 6]
 [1 0 6 3 8 0 4 1 5 5]
 [9 1 3 1 8 2 7 5 1 6]
 [1 1 7 8 7 4 4 1 3 4]
 [7 7 8 1 8 6 4 4 2 8]
 [5 8 9 6 7 4 1 7 5 1]
 [0 6 5 8 6 9 8 4 1 5]]
</pre>
We split the assignment into 2 sections.

Section 1

</div>
<div class="column">
28. It is split into training set of 60,000 examples, and a test set of 10,000

</div>
</div>
<div class="layoutArea">
<div class="column">
In [189]:

In [190]:

In [192]:

In [193]:

In [195]:

In [196]:

In [197]:

In [198]:

In [200]:

In [114]:

In [202]:

In [203]:

In [117]:

In [205]:

In [207]:

In [208]:

In [209]:

In [210]:

In [212]:

In [213]:

In [214]:

In [215]:

In [218]:

In [219]:

In [220]:

</div>
<div class="column">
We will define the activation functions and their derivatives which will be used later during forward and backward propagation. We will define the softmax cross entropy loss for calculating the prediction loss.

Section 2

We will initialize the network and define forward and backward propagation through a single layer. We will extend this to multiple layers of a network. We will initialize and train the multi-layer neural network

Section 1 Activation Functions

An Activation function usually adds nonlinearity to the output of a network layer using a mathematical operation. We will use two types of activation function in this assignment,

Rectified Linear Unit or ReLU Linear activation (This is a dummy activation function without any nonlinearity implemented for convenience)

ReLU (Rectified Linear Unit) (10 points)

ReLU (Rectified Linear Unit) is a piecewise linear function defined as Hint: use numpy.maximum

ReLU – Gradient (15 points)

The gradient of ReLu( ) is 1 if else it is 0.

Linear Activation

There is no activation involved here. It is an identity function.

Softmax Activation and Cross-entropy Loss Function (15 Points)

The softmax activation is computed on the outputs from the last layer and the output label with the maximum probablity is predicted as class label. The

</div>
</div>
<div class="layoutArea">
<div class="column">
In [ ]:

</div>
<div class="column">
<pre># Contains hidden tests testing for test data accuracy &gt; 85%
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>#import libraries and functions to load the data
</pre>
from digits import get_mnist

from matplotlib import pyplot as plt import numpy as np

import ast

import sys

import numpy.testing as npt

import pytest

import random

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>random.seed(1)
np.random.seed(1)
trX, trY, tsX, tsY = get_mnist()
print('trX.shape: ', trX.shape)
print('trY.shape: ', trY.shape)
print('tsX.shape: ', tsX.shape)
print('tsY.shape: ', tsY.shape)
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre># The data is of the format (no_samples, channels, img_height, img_width)
# In the training data trX, there are 60000 images. Each image has one channel (gray scale).
# Each image is of height=28 and width=28 pixels
# Lets sample a smaller subest to work with.
# We will use 2000 training examples and 1000 test samples.
# We define a function which we can use later as well
</pre>
def sample_mnist(n_train=2000, n_test=1000): trX, trY, tsX, tsY = get_mnist() random.seed(1)

np.random.seed(1)

<pre>    tr_idx = np.random.choice(trX.shape[0], n_train)
    trX = trX[tr_idx]
    trY = trY[tr_idx]
    ts_idx = np.random.choice(tsX.shape[0], n_train)
    tsX = tsX[ts_idx]
</pre>
tsY = tsY[ts_idx]

trX = trX.reshape(-1, 28*28).T trY = trY.reshape(1, -1)

tsX = tsX.reshape(-1, 28*28).T tsY = tsY.reshape(1, -1) return trX, trY, tsX, tsY

<pre># Lets verify the function
</pre>
trX, trY, tsX, tsY = sample_mnist(n_train=2000, n_test=1000) # Lets examine the data and see if it is normalized print(‘trX.shape: ‘, trX.shape)

print(‘trY.shape: ‘, trY.shape)

print(‘tsX.shape: ‘, tsX.shape)

print(‘tsY.shape: ‘, tsY.shape)

print(‘Train max: value = {}, Train min: value = {}’.format(np.max(trX), np.min(trX))) print(‘Test max: value = {}, Test min: value = {}’.format(np.max(tsX), np.min(tsX))) print(‘Unique labels in train: ‘, np.unique(trY))

print(‘Unique labels in test: ‘, np.unique(tsY))

<pre># Let's visualize a few samples and their labels from the train and test datasets.
</pre>
print(‘\nDisplaying a few samples’)

visx = np.concatenate((trX[:,:50],tsX[:,:50]), axis=1).reshape(28,28,10,10).transpose(2,0,3,1).reshape(28*10,-1) visy = np.concatenate((trY[:,:50],tsY[:,:50]), axis=1).reshape(10,-1)

print(‘labels’)

print(visy)

plt.figure(figsize = (8,8))

plt.axis(‘off’)

plt.imshow(visx, cmap=’gray’);

</div>
</div>
<div class="layoutArea">
<div class="column">
def relu(Z): ”’

<pre>    Computes relu activation of input Z
</pre>
<pre>    Inputs:
        Z: numpy.ndarray (n, m) which represent 'm' samples each of 'n' dimension
</pre>
<pre>    Outputs:
        A: where A = ReLU(Z) is a numpy.ndarray (n, m) representing 'm' samples each of 'n' dimension
        cache: a dictionary with {"Z", Z}
</pre>
”’

cache = {}

# your code here

<pre>    A = np.maximum(0,Z)
    cache["Z"]=Z
</pre>
return A, cache

</div>
</div>
<div class="layoutArea">
<div class="column">
#Test

<pre>z_tst = [-2,-1,0,1,2]
a_tst, c_tst = relu(z_tst)
npt.assert_array_equal(a_tst,[0,0,0,1,2])
npt.assert_array_equal(c_tst["Z"], [-2,-1,0,1,2])
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
def relu_der(dA, cache): ”’

<pre>    Computes derivative of relu activation
</pre>
<pre>    Inputs:
        dA: derivative from the subsequent layer of dimension (n, m).
</pre>
<pre>            dA is multiplied elementwise with the gradient of ReLU
        cache: dictionary with {"Z", Z}, where Z was the input
            to the activation layer during forward propagation
</pre>
<pre>    Outputs:
        dZ: the derivative of dimension (n,m). It is the elementwise
</pre>
<pre>            product of the derivative of ReLU and dA
</pre>
”’

dZ = np.array(dA, copy=True) Z = cache[“Z”]

# your code here

for i in range(Z.shape[0]):

for j in range(Z.shape[1]): if Z[i][j] &gt; 0:

dZ[i][j]=dA[i][j] else:

</div>
</div>
<div class="layoutArea">
<div class="column">
return dZ

</div>
</div>
<div class="layoutArea">
<div class="column">
dZ[i][j]=0

</div>
</div>
<div class="layoutArea">
<div class="column">
#Test

<pre>dA_tst = np.array([[0,2],[1,1]])
cache_tst = {}
cache_tst['Z'] = np.array([[-1,2],[1,-2]])
npt.assert_array_equal(relu_der(dA_tst,cache_tst),np.array([[0,2],[1,0]]))
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
def linear(Z): ”’

<pre>    Computes linear activation of Z
    This function is implemented for completeness
</pre>
<pre>    Inputs:
        Z: numpy.ndarray (n, m) which represent 'm' samples each of 'n' dimension
</pre>
<pre>    Outputs:
        A: where A = Linear(Z) is a numpy.ndarray (n, m) representing 'm' samples each of 'n' dimension
        cache: a dictionary with {"Z", Z}
</pre>
”’

A=Z

cache = {} cache[“Z”] = Z return A, cache

</div>
</div>
<div class="layoutArea">
<div class="column">
def linear_der(dA, cache): ”’

<pre>    Computes derivative of linear activation
    This function is implemented for completeness
</pre>
<pre>    Inputs:
        dA: derivative from the subsequent layer of dimension (n, m).
</pre>
<pre>            dA is multiplied elementwise with the gradient of Linear(.)
        cache: dictionary with {"Z", Z}, where Z was the input
            to the activation layer during forward propagation
</pre>
<pre>    Outputs:
        dZ: the derivative of dimension (n,m). It is the elementwise
</pre>
<pre>            product of the derivative of Linear(.) and dA
</pre>
”’

dZ = np.array(dA, copy=True) return dZ

</div>
</div>
<div class="layoutArea">
<div class="column">
softmax function can also be refered as normalized exponential function which takes a vector of distribution consisting of probabilities proportional to the exponentials of the input numbers.

</div>
<div class="column">
real numbers as input, and normalizes it into a probability

is the sample of dimensions. We estimate the softmax , where the components of are,

</div>
</div>
<div class="layoutArea">
<div class="column">
The input to the softmax function is the matrix,

for each of the samples to . The softmax activation for sample

</div>
<div class="column">
, where

</div>
</div>
<div class="layoutArea">
<div class="column">
The output of the softmax is from all the input components of

If the output of softmax is given by groundtruth labels is given by,

</div>
<div class="column">
, where

before calculating the softmax. This constant is

and the ground truth is given by

</div>
<div class="column">
. In order to avoid floating point overflow, we subtract a constant , where, . The activation is given by,

, the cross entropy loss between the predictions and

</div>
</div>
<div class="layoutArea">
<div class="column">
is

</div>
</div>
<div class="layoutArea">
<div class="column">
where is the identity function given by

</div>
</div>
<div class="layoutArea">
<div class="column">
Hint: use numpy.exp numpy.max, numpy.sum numpy.log Also refer to use of ‘keepdims’ and ‘axis’ parameter.

</div>
</div>
<div class="layoutArea">
<div class="column">
def softmax_cross_entropy_loss(Z, Y=np.array([])): ”’

<pre>    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss
</pre>
<pre>    Inputs:
        Z: numpy.ndarray (n, m)
        Y: numpy.ndarray (1, m) of labels
</pre>
<pre>            when y=[] loss is set to []
</pre>
<pre>    Outputs:
        A: numpy.ndarray (n, m) of softmax activations
        cache: a dictionary to store the activations which will be used later to estimate derivatives
        loss: cost of prediction
</pre>
”’

<pre>    # your code here
</pre>
A=np.copy(Z)

if (Y.size == 0):

loss = [] else:

<pre>        loss = 0
    m = Z.shape[1]
</pre>
for col in range(Z.shape[1]):

sum_exp = np.sum(np.exp(Z[:,col])) for row in range(Z.shape[0]):

A[row][col]=np.exp(Z[row][col])/sum_exp if (Y.size!=0 and Y[0][col]==row):

loss = loss + np.log(A[row][col]) if (Y.size != 0):

<pre>        loss = -1/m * loss
</pre>
cache = {} cache[“A”] = A return A, cache, loss

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>#test cases for softmax_cross_entropy_loss
</pre>
<pre>np.random.seed(1)
Z_t = np.random.randn(3,4)
Y_t = np.array([[1,0,1,2]])
A_t = np.array([[0.57495949, 0.38148818, 0.05547572, 0.36516899],
</pre>
<pre>       [0.26917503, 0.07040735, 0.53857622, 0.49875847],
       [0.15586548, 0.54810447, 0.40594805, 0.13607254]])
</pre>
<pre>A_est, cache_est, loss_est = softmax_cross_entropy_loss(Z_t, Y_t)
npt.assert_almost_equal(loss_est,1.2223655548779273,decimal=5)
npt.assert_array_almost_equal(A_est,A_t,decimal=5)
npt.assert_array_almost_equal(cache_est['A'],A_t,decimal=5)
</pre>
<pre># hidden test cases follow
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
Derivative of the softmax_cross_entropy_loss(.) (15 points)

We discused in the lecture that it is easier to directly estimate which is , where

Let be the dimension input and be the groundtruth labels. If given by,

where, is the one-hot representation of .

</div>
<div class="column">
is the input to the softmax_cross_entropy_loss(

</div>
</div>
<div class="layoutArea">
<div class="column">
One-hot encoding is a binary representation of the discrete class labels. For example, let data points. In this case will be a matrix. Let the categories of the 4 data points be

</div>
<div class="column">
for a 3-category problem. Assume there are . The one hot representation is given by,

is

</div>
</div>
<div class="layoutArea">
<div class="column">
where, the one-hot encoding for label

Section 2

Parameter Initialization (10 points)

</div>
<div class="column">
is . Similarly, the one-hot encoding for

</div>
</div>
<div class="layoutArea">
<div class="column">
def softmax_cross_entropy_loss_der(Y, cache): ”’

<pre>    Computes the derivative of the softmax activation and cross entropy loss
</pre>
<pre>    Inputs:
        Y: numpy.ndarray (1, m) of labels
        cache: a dictionary with cached activations A of size (n,m)
</pre>
<pre>    Outputs:
        dZ: derivative dL/dZ - a numpy.ndarray of dimensions (n, m)
</pre>
”’

<pre>    Y_hot=np. array([])
    A = cache["A"]
    dZ = np.copy(A)
    m = Y.shape[1]
</pre>
for col in range(A.shape[1]):

for row in range(A.shape[0]):

if (Y[0][col]==row):

dZ[row][col]= 1/m * (A[row][col] – 1)

else:

dZ[row][col]= 1/m * A[row][col]

<pre>    # your code here
</pre>
return dZ

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>#test cases for softmax_cross_entropy_loss_der
</pre>
<pre>np.random.seed(1)
Z_t = np.random.randn(3,4)
Y_t = np.array([[1,0,1,2]])
A_t = np.array([[0.57495949, 0.38148818, 0.05547572, 0.36516899],
</pre>
<pre>       [0.26917503, 0.07040735, 0.53857622, 0.49875847],
</pre>
<pre>       [0.15586548, 0.54810447, 0.40594805, 0.13607254]])
cache_t={}
</pre>
<pre>cache_t['A'] = A_t
dZ_t = np.array([[ 0.14373987, -0.15462795,  0.01386893,  0.09129225],
</pre>
<pre>       [-0.18270624,  0.01760184, -0.11535594,  0.12468962],
       [ 0.03896637,  0.13702612,  0.10148701, -0.21598186]])
</pre>
<pre>dZ_est = softmax_cross_entropy_loss_der(Y_t, cache_t)
npt.assert_almost_equal(dZ_est,dZ_t,decimal=5)
</pre>
<pre># hidden test cases follow
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
Let us now define a function that can initialize the parameters of the multi-layer neural network. The network parameters will be stored as dictionary elements that can easily be passed as function parameters while calculating gradients during back propogation.

1. The weight matrix is initialized with random values from a normal distribution with variance . For example, to create a matrix of dimension , with

</div>
</div>
<div class="layoutArea">
<div class="column">
values from a normal distribution with variance , we write

zero for faster training.

2. Bias values are initialized with 0. For example a bias vector of dimensions is initialized as

The dimension for weight matrix for layer is given by ( Number-of-neurons-in-layer- bias for for layer is (Number-of-neurons-in-layer- 1)

def initialize_network(net_dims): ”’

<pre>    Initializes the parameters of a multi-layer neural network
</pre>
</div>
<div class="column">
. The is to ensure very small values close to

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>    Inputs:
        net_dims: List containing the dimensions of the network. The values of the array represent the number of nod
</pre>
<pre>es in
        each layer. For Example, if a Neural network contains 784 nodes in the input layer, 800 in the first hidden
</pre>
<pre> layer,
        500 in the secound hidden layer and 10 in the output layer, then net_dims = [784,800,500,10].
</pre>
<pre>    Outputs:
        parameters: Python Dictionary for storing the Weights and bias of each layer of the network
</pre>
”’

numLayers = len(net_dims) parameters = {}

for l in range(numLayers-1):

# Hint:

# parameters[“W”+str(l+1)] =

# parameters[“b”+str(l+1)] =

# your code here

dim_current_layer = net_dims[l+1]

dim_previous_layer = net_dims[l]

parameters[“W”+str(l+1)] = 0.01 * np.random.randn(dim_current_layer, dim_previous_layer) parameters[“b”+str(l+1)] = np.zeros((dim_current_layer,1))

</div>
</div>
<div class="layoutArea">
<div class="column">
return parameters

#Test

net_dims_tst = [5,4,1]

parameters_tst = initialize_network(net_dims_tst) assert parameters_tst[‘W1’].shape == (4,5)

assert parameters_tst[‘W2’].shape == (1,4)

assert parameters_tst[‘b1’].shape == (4,1)

assert parameters_tst[‘b2’].shape == (1,1)

assert parameters_tst[‘b1’].all() == 0

assert parameters_tst[‘b2’].all() == 0

# There are hidden tests

Forward Propagation Through a Single Layer (5 points)

</div>
</div>
<div class="layoutArea">
<div class="column">
If the vectorized input to any layer of neural network is activation is):

def linear_forward(A_prev, W, b): ”’

<pre>    Input A_prev propagates through the layer
    Z = WA + b is the output of this layer.
</pre>
</div>
<div class="column">
and the parameters of the layer is given by

</div>
<div class="column">
, the output of the layer (before the

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>    Inputs:
        A_prev: numpy.ndarray (n,m) the input to the layer
        W: numpy.ndarray (n_out, n) the weights of the layer
        b: numpy.ndarray (n_out, 1) the bias of the layer
</pre>
<pre>    Outputs:
        Z: where Z = W.A_prev + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache: a dictionary containing the inputs A
</pre>
”’

<pre>    # your code here
</pre>
<pre>    Z = np.dot(W, A_prev) + b
</pre>
cache = {} cache[“A”] = A_prev return Z, cache

<pre>#Hidden test cases follow
</pre>
<pre>np.random.seed(1)
n1 = 3
m1 = 4
A_prev_t = np.random.randn(n1,m1)
W_t = np.random.randn(n1, n1)
</pre>
<pre>b_t = np.random.randn(n1, 1)
Z_est, cache_est = linear_forward(A_prev_t, W_t, b_t)
</pre>
Activation After Forward Propagation

The linear transformation in a layer is usually followed by a nonlinear activation function given by,

Depending on the activation choosen for the given layer, the can represent different operations.

def layer_forward(A_prev, W, b, activation): ”’

<pre>    Input A_prev propagates through the layer and the activation
</pre>
<pre>    Inputs:
        A_prev: numpy.ndarray (n,m) the input to the layer
        W: numpy.ndarray (n_out, n) the weights of the layer
        b: numpy.ndarray (n_out, 1) the bias of the layer
        activation: is the string that specifies the activation function
</pre>
<pre>    Outputs:
        A: = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache: a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
</pre>
”’

Z, lin_cache = linear_forward(A_prev, W, b) if activation == “relu”:

A, act_cache = relu(Z) elif activation == “linear”: A, act_cache = linear(Z)

cache = {}

cache[“lin_cache”] = lin_cache cache[“act_cache”] = act_cache return A, cache

Multi-Layers Forward Propagation

Multiple layers are stacked to form a multi layer network. The number of layers in the network can be inferred from the size of the initialize_network() function. If the number of items in the dictionary element is , then the number of layers will be

</div>
<div class="column">
variable from

</div>
</div>
<div class="layoutArea">
<div class="column">
During forward propagation, the input which is a matrix of samples where each sample is dimensions, is input into the first layer. The subsequent layers use the activation output from the previous layer as inputs.

</div>
</div>
<div class="layoutArea">
<div class="column">
Note all the hidden layers in our network use ReLU activation except the last layer which uses Linear activation. Forward Propagation

def multi_layer_forward(A0, parameters): ”’

<pre>    Forward propgation through the layers of the network
</pre>
<pre>    Inputs:
        A0: numpy.ndarray (n,m) with n features and m samples
        parameters: dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
</pre>
<pre>    Outputs:
        AL: numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
</pre>
<pre>            where c is number of categories and m is number of samples
        caches: a dictionary of associated caches of parameters and network inputs
</pre>
”’

<pre>    L = len(parameters)//2
    A = A0
    caches = []
</pre>
for l in range(1,L):

A, cache = layer_forward(A, parameters[“W”+str(l)], parameters[“b”+str(l)], “relu”) caches.append(cache)

AL, cache = layer_forward(A, parameters[“W”+str(L)], parameters[“b”+str(L)], “linear”) caches.append(cache)

return AL, caches

Backward Propagagtion Through a Single Layer (10 points)

</div>
</div>
<div class="layoutArea">
<div class="column">
Consider the linear layer represented as

</div>
<div class="column">
. We would like to estimate the gradients – represented as , – represented as . The input to estimate these derivatives is – represented as . The derivatives are given by,

</div>
<div class="column">
and –

is of

</div>
</div>
<div class="layoutArea">
<div class="column">
where dimensions

</div>
<div class="column">
is matrix of derivatives. The figure below represents a case fo binary cassification where . The example can be extended to .

</div>
</div>
<div class="layoutArea">
<div class="column">
Backward Propagation

def linear_backward(dZ, cache, W, b): ”’

<pre>    Backward prpagation through the linear layer
</pre>
<pre>    Inputs:
        dZ: numpy.ndarray (n,m) derivative dL/dz
        cache: a dictionary containing the inputs A, for the linear layer
</pre>
<pre>            where Z = WA + b,
</pre>
<pre>            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W: numpy.ndarray (n,p)
        b: numpy.ndarray (n, 1)
</pre>
<pre>    Outputs:
        dA_prev: numpy.ndarray (p,m) the derivative to the previous layer
        dW: numpy.ndarray (n,p) the gradient of W
        db: numpy.ndarray (n, 1) the gradient of b
</pre>
”’

A = cache[“A”]

# your code here

dA_prev = np.dot(np.transpose(W), dZ) dW = np.dot(dZ, np.transpose(A))

db = np.sum(dZ, axis=1, keepdims=True) return dA_prev, dW, db

<pre>#Hidden test cases follow
</pre>
<pre>np.random.seed(1)
n1 = 3
m1 = 4
p1 = 5
</pre>
<pre>dZ_t = np.random.randn(n1,m1)
A_t = np.random.randn(p1,m1)
cache_t = {}
cache_t['A'] = A_t
</pre>
<pre>W_t = np.random.randn(n1,p1)
b_t = np.random.randn(n1,1)
</pre>
dA_prev_est, dW_est, db_est = linear_backward(dZ_t, cache_t, W_t, b_t) Back Propagation With Activation

We will define the backpropagation for a layer. We will use the backpropagation for a linear layer along with the derivative for the activation.

def layer_backward(dA, cache, W, b, activation): ”’

<pre>    Backward propagation through the activation and linear layer
</pre>
<pre>    Inputs:
        dA: numpy.ndarray (n,m) the derivative to the previous layer
        cache: dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W: numpy.ndarray (n,p)
        b: numpy.ndarray (n, 1)
</pre>
<pre>    Outputs:
        dA_prev: numpy.ndarray (p,m) the derivative to the previous layer
        dW: numpy.ndarray (n,p) the gradient of W
        db: numpy.ndarray (n, 1) the gradient of b
</pre>
”’

<pre>    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]
</pre>
if activation == “relu”:

dZ = relu_der(dA, act_cache)

elif activation == “linear”:

dZ = linear_der(dA, act_cache)

dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b) return dA_prev, dW, db

Multi-layers Back Propagation

</div>
</div>
<div class="layoutArea">
<div class="column">
We have defined the required functions to handle back propagation for single layer. Now we will stack the layers together and perform back propagation on the entire network.

def multi_layer_backward(dAL, caches, parameters): ”’

<pre>    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately
</pre>
<pre>    Inputs:
        dAL: numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches: a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
</pre>
<pre>    Outputs:
        gradients: dictionary of gradient of network parameters
</pre>
<pre>            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
</pre>
”’

L = len(caches)

gradients = {}

dA = dAL

activation = “linear”

for l in reversed(range(1,L+1)):

<pre>        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward(dA, caches[l-1], \
</pre>
<pre>                    parameters["W"+str(l)],parameters["b"+str(l)],\
</pre>
<pre>                    activation)
        activation = "relu"
</pre>
return gradients

Prediction (10 points)

We will perform forward propagation through the entire network and determine the class predictions for the input data

</div>
</div>
<div class="layoutArea">
<div class="column">
def

#

</div>
<div class="column">
classify(X, parameters):

”’

Network prediction for inputs X

<pre>Inputs:
    X: numpy.ndarray (n,m) with n features and m samples
    parameters: dictionary of network parameters
</pre>
<pre>        {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
Outputs:
</pre>
<pre>    YPred: numpy.ndarray (1,m) of predictions
'''
</pre>
<pre># Forward propagate input 'X' using multi_layer_forward(.) and obtain the final activation 'A'
# Using 'softmax_cross_entropy loss(.)', obtain softmax activation 'AL' with input 'A' from step 1
# Predict class label 'YPred' as the 'argmax' of softmax activation from step-2.
# Note: the shape of 'YPred' is (1,m), where m is the number of samples
</pre>
<pre># your code here
</pre>
<pre>A, caches = multi_layer_forward(X, parameters)
AL,cache,loss = softmax_cross_entropy_loss(A)
YPred = np.array([])
</pre>
<pre>  YPred = np.reshape(YPred, (-1,1))
</pre>
YPred = np.argmax(AL, axis=0)

YPred = YPred.reshape(-1, YPred.size) return YPred

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>#Hidden test cases follow
</pre>
<pre>np.random.seed(1)
n1 = 3
m1 = 4
p1 = 2
</pre>
<pre>X_t = np.random.randn(n1,m1)
W1_t = np.random.randn(p1,n1)
b1_t = np.random.randn(p1,1)
W2_t = np.random.randn(p1,p1)
b2_t = np.random.randn(p1,1)
parameters_t = {'W1':W1_t, 'b1':b1_t, 'W2':W2_t, 'b2':b2_t}
YPred_est = classify(X_t, parameters_t)
</pre>
Parameter Update Using Batch-Gradient

The parameter gradients calculated during back propagation are used to update the values of the network parameters.

where is the learning rate of the network.

def update_parameters(parameters, gradients, epoch, alpha): ”’

<pre>    Updates the network parameters with gradient descent
</pre>
<pre>    Inputs:
        parameters: dictionary of network parameters
</pre>
<pre>            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients: dictionary of gradient of network parameters
</pre>
<pre>            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch: epoch number
        alpha: step size or learning rate
</pre>
<pre>    Outputs:
        parameters: updated dictionary of network parameters
</pre>
<pre>            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
</pre>
”’

L = len(parameters)//2 for i in range(L):

#parameters[“W”+str(i+1)] =

#parameters[“b”+str(i+1)] =

# your code here

parameters[“W”+str(i+1)] = parameters[“W”+str(i+1)] – alpha * gradients[“dW”+str(i+1)] parameters[“b”+str(i+1)] = parameters[“b”+str(i+1)] – alpha * gradients[“db”+str(i+1)]

return parameters Neural Network

Let us now assemble all the components of the neural network together and define a complete training loop for a Multi-layer Neural Network.

def multi_layer_network(X, Y, net_dims, num_iterations=500, learning_rate=0.1, log=True): ”’

<pre>    Creates the multilayer network and trains the network
</pre>
<pre>    Inputs:
        X: numpy.ndarray (n,m) of training data
        Y: numpy.ndarray (1,m) of training data labels
        net_dims: tuple of layer dimensions
        num_iterations: num of epochs to train
        learning_rate: step size for gradient descent
        log: boolean to print training progression
</pre>
<pre>    Outputs:
        costs: list of costs (or loss) over training
        parameters: dictionary of trained network parameters
</pre>
”’

<pre>    parameters = initialize_network(net_dims)
    A0 = X
    costs = []
    num_classes = 10
</pre>
alpha = learning_rate

for ii in range(num_iterations):

<pre>        ## Forward Propagation
        # Step 1: Input 'A0' and 'parameters' into the network using multi_layer_forward()
        #         and calculate output of last layer 'A' (before softmax) and obtain cached activations as 'caches'
        # Step 2: Input 'A' and groundtruth labels 'Y' to softmax_cros_entropy_loss(.) and estimate
        #         activations 'AL', 'softmax_cache' and 'loss'
</pre>
<pre>        ## Back Propagation
        # Step 3: Estimate gradient 'dAL' with softmax_cros_entropy_loss_der(.) using groundtruth
        #         labels 'Y' and 'softmax_cache'
        # Step 4: Estimate 'gradients' with multi_layer_backward(.) using 'dAL' and 'parameters'
        # Step 5: Estimate updated 'parameters' and updated learning rate 'alpha' with update_parameters(.)
</pre>
<ul>
<li>
<pre>        # &nbsp;        using 'parameters', 'gradients', loop variable 'ii' (epoch number) and 'learning_rate'
</pre>
</li>
<li>
<pre>        # &nbsp;        Note: Use the same variable 'parameters' as input and output to the update_parameters(.) function
</pre>
<pre>        # your code here
</pre>
<pre>        A, caches = multi_layer_forward(A0, parameters)
        AL, softmax_cache, cost = softmax_cross_entropy_loss(A, Y)
        dAL = softmax_cross_entropy_loss_der(Y, softmax_cache)
        gradients = multi_layer_backward(dAL, caches, parameters)
        parameters = update_parameters(parameters, gradients, ii, learning_rate)
</pre>
if ii % 20 == 0: costs.append(cost) if log:

print(“Cost at iteration %i is: %.05f, learning rate: %.05f” %(ii+1, cost, learning_rate)) return costs, parameters

Training – 10 points

We will now intialize a neural network with 1 hidden layer whose dimensions is 200. Since the input samples are of dimension 28 28, the input layer will be of dimension 784. The output dimension is 10 since we have a 10 category classification. We will train the model and compute its accuracy on both training and test sets and plot the training cost (or loss) against the number of iterations.

<pre># You should be able to get a train accuracy of &gt;90% and a test accuracy &gt;85%
# The settings below gave &gt;95% train accuracy and &gt;90% test accuracy
# Feel free to adjust the values and explore how the network behaves
</pre>
net_dims = [784,200,10]

#784 is for image dimensions #10 is for number of categories #200 is arbitrary

<pre># initialize learning rate and num_iterations
</pre>
<pre>learning_rate = 0.1
num_iterations = 500
</pre>
<pre>np.random.seed(1)
print("Network dimensions are:" + str(net_dims))
</pre>
<pre># getting the subset dataset from MNIST
</pre>
<pre>trX, trY, tsX, tsY = sample_mnist(n_train=2000, n_test=1000)
costs, parameters = multi_layer_network(trX, trY, net_dims, \
</pre>
num_iterations=num_iterations, learning_rate=learning_rate) # compute the accuracy for training set and testing set

<pre>train_Pred = classify(trX, parameters)
test_Pred = classify(tsX, parameters)
</pre>
#Estimate the training accuracy ‘trAcc’ and the testing accuracy ‘teAcc’ # your code here

if trY.size!=0:

trAcc = np.mean(train_Pred==trY) if tsY.size!=0:

teAcc = np.mean(test_Pred==tsY)

print(“Accuracy for training set is {0:0.3f} %”.format(trAcc))

print(“Accuracy for testing set is {0:0.3f} %”.format(teAcc))

<pre>plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
</pre>
<pre>Network dimensions are:[784, 200, 10]
Cost at iteration 1 is: 2.30262, learning rate: 0.10000
Cost at iteration 21 is: 1.80242, learning rate: 0.10000
Cost at iteration 41 is: 0.92838, learning rate: 0.10000
Cost at iteration 61 is: 0.71804, learning rate: 0.10000
Cost at iteration 81 is: 0.58955, learning rate: 0.10000
Cost at iteration 101 is: 0.53325, learning rate: 0.10000
Cost at iteration 121 is: 0.43405, learning rate: 0.10000
Cost at iteration 141 is: 0.42289, learning rate: 0.10000
Cost at iteration 161 is: 0.34212, learning rate: 0.10000
Cost at iteration 181 is: 0.33355, learning rate: 0.10000
Cost at iteration 201 is: 0.31770, learning rate: 0.10000
Cost at iteration 221 is: 0.28235, learning rate: 0.10000
Cost at iteration 241 is: 0.27505, learning rate: 0.10000
Cost at iteration 261 is: 0.24805, learning rate: 0.10000
Cost at iteration 281 is: 0.23275, learning rate: 0.10000
Cost at iteration 301 is: 0.22678, learning rate: 0.10000
Cost at iteration 321 is: 0.20941, learning rate: 0.10000
Cost at iteration 341 is: 0.20796, learning rate: 0.10000
Cost at iteration 361 is: 0.18465, learning rate: 0.10000
Cost at iteration 381 is: 0.17449, learning rate: 0.10000
Cost at iteration 401 is: 0.17399, learning rate: 0.10000
Cost at iteration 421 is: 0.15403, learning rate: 0.10000
Cost at iteration 441 is: 0.14431, learning rate: 0.10000
Cost at iteration 461 is: 0.14161, learning rate: 0.10000
Cost at iteration 481 is: 0.13169, learning rate: 0.10000
Accuracy for training set is 0.976 %
Accuracy for testing set is 0.902 %
</pre>
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
is the

</div>
<div class="column">
) function.

matrix of softmax activations of , the derivative is

</div>
</div>
<div class="layoutArea">
<div class="column">
Number-of-neurons-in-layer-

</div>
<div class="column">
). The dimension of the

</div>
</div>
</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
4 = m

</div>
<div class="column">
}2,1,0{ ∈ )i(y

</div>
</div>
<div class="layoutArea">
<div class="column">
A

</div>
</div>
<div class="layoutArea">
<div class="column">
Zd

</div>
<div class="column">
)m × n(

</div>
<div class="column">
])m(zd,…,)2(zd,)1(zd[ = Zd

</div>
</div>
<div class="layoutArea">
<div class="column">
L sretemarap

</div>
<div class="column">
L2 sretemarap

</div>
</div>
<div class="layoutArea">
<div class="column">
×

</div>
</div>
<div class="layoutArea">
<div class="column">
verp_Ad bd

Ld Ld Ld

</div>
</div>
<div class="layoutArea">
<div class="column">
l

</div>
<div class="column">
× )1+l( ))4 ,3((sorez .pn = b

</div>
<div class="column">
)1+l(

</div>
</div>
<div class="layoutArea">
<div class="column">
4×3w1

</div>
</div>
<div class="layoutArea">
<div class="column">
⊤]1,0,0[ = )4( ̄y

</div>
<div class="column">
2 = )4(y ]2,1,0,1[= Y

</div>
<div class="column">
⎦1000⎣

⎥ 0 1 0 1 ⎢ = ̄Y ⎤0010⎡

</div>
<div class="column">
⊤]0,1,0[ = )1( ̄y

</div>
<div class="column">
1 = )1(y

4×3 Z

</div>
</div>
<div class="layoutArea">
<div class="column">
bd Wd

</div>
<div class="column">
Wd

</div>
</div>
<div class="layoutArea">
<div class="column">
)b , W(

</div>
<div class="column">
verp_A

</div>
</div>
<div class="layoutArea">
<div class="column">
n≤k≤1 rof )nz…,2z,1z(xam = xamz

</div>
<div class="column">
)i( )xamz −

</div>
<div class="column">
n =)i(ka kz(pxe

</div>
</div>
<div class="layoutArea">
<div class="column">
)i(a

n

</div>
<div class="column">
))i(z(xamtfos = )i(a )i(z

</div>
<div class="column">
m 1 )m × n(

</div>
</div>
<div class="layoutArea">
<div class="column">
n

</div>
<div class="column">
m m×n 0A

</div>
</div>
<div class="layoutArea">
<div class="column">
hti )i(z n

</div>
<div class="column">
])m(z,…,)2(z,)1(z[ = Z

</div>
</div>
<div class="layoutArea">
<div class="column">
Zd

</div>
<div class="column">
Zd Ld

</div>
<div class="column">
verp_Ad b+verp_A.W=Z

</div>
</div>
<div class="layoutArea">
<div class="column">
1 × 3

10.0 )4 ,3(ndnar .modnar .pn ∗ 10.0 = w

</div>
<div class="column">
1

</div>
</div>
<div class="layoutArea">
<div class="column">
,bd.α−b=:b Wd .α − W =: W

</div>
</div>
<div class="layoutArea">
<div class="column">
1=i )i(Zd∑ = bd

</div>
</div>
<div class="layoutArea">
<div class="column">
m TAZd = Wd

</div>
</div>
<div class="layoutArea">
<div class="column">
Zd T W = verp_Ad

</div>
</div>
<div class="layoutArea">
<div class="column">
) .(σ .)Z(σ = A

</div>
</div>
<div class="layoutArea">
<div class="column">
b + verp_A . W = Z

</div>
</div>
<div class="layoutArea">
<div class="column">
b + verp_A . W = Z

</div>
</div>
<div class="layoutArea">
<div class="column">
) ̄Y − A ( m = Z d 1

</div>
</div>
<div class="layoutArea">
<div class="column">
ZdZ )m×n(A)m,1(Y)m×n(Z Z ZZdZd

</div>
</div>
<div class="layoutArea">
<div class="column">
eslaF = noitidnoc fi ,0 = }noitidnoc{I eurT = noitidnoc fi ,1 = }noitidnoc{I

</div>
</div>
<div class="layoutArea">
<div class="column">
1=k 1=i m

ki a g o l } k = i y { I ∑ ∑ 1 − = ) Y , A ( s s o L

</div>
</div>
<div class="layoutArea">
<div class="column">
)i(

⊤] na,…, 2a, 1a[ = )i(a

</div>
<div class="column">
)i(z ])m(a….)2(a,)1(a[ = A

</div>
</div>
<div class="layoutArea">
<div class="column">
xamz

)i( )i( )i(

</div>
</div>
<div class="layoutArea">
<div class="column">
) kz(pxe1=k∑ n≤k≤1 rof )i( n =)i(ka

</div>
</div>
<div class="layoutArea">
<div class="column">
Z = )Z(raeniL

</div>
</div>
<div class="layoutArea">
<div class="column">
)x,0(xam = )x(ULeR

</div>
</div>
<div class="layoutArea">
<div class="column">
Ld

</div>
</div>
<div class="layoutArea">
<div class="column">
nm

])m(y,…,)2(y,)1(y[ = Y A

</div>
<div class="column">
Y

</div>
</div>
<div class="layoutArea">
<div class="column">
)xamz − kz(pxe 1=k∑

</div>
</div>
<div class="layoutArea">
<div class="column">
) kz(pxe )i(

</div>
</div>
<div class="layoutArea">
<div class="column">
×

</div>
</div>
<div class="layoutArea">
<div class="column">
)m×n( )m×1(

</div>
</div>
<div class="layoutArea">
<div class="column">
× )1+l(

</div>
<div class="column">
)1+l(

</div>
</div>
<div class="layoutArea">
<div class="column">
)bd , Wd(

</div>
</div>
<div class="layoutArea">
<div class="column">
Y ̄Y

</div>
</div>
<div class="layoutArea">
<div class="column">
0&gt;Z Z

</div>
</div>
<div class="layoutArea">
<div class="column">
n

</div>
</div>
<div class="layoutArea">
<div class="column">
α

</div>
</div>
<div class="layoutArea">
<div class="column">
I

</div>
</div>
</div>
</div>

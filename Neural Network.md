# NEURAL NETWORK

- y=f(x)
- 3-layer NN : y = f3(f2(f1(x)))
	- where f1,f2 : f(z) = g(Wz+b)
	- g : ACTIVATION FUNCTION (hyperparameter) - fixed and non-linear
	- g can be sigmoid to have logistic regression
	- f3 : scalar function / vector function
	- W : for each layer (same dimensionality as z) ; w for each neuron in each layer

## Multilayer Perceptron / Vanilla Neural Network

- Eg: 3 layer FFNN -> classification / regression depending upon third output layer activation function
- At each neuron, :
	- all inputs of the unit are joined together to form input vector
	- linear transformation to the input vector
	- applies an activation function g to the result of the linear transformation
	- outputs a real value
	- in ffnn, output of few neurons become input of subsequent layer
	- may contain Fully connected layers.

## FFNN

- For regression / classification : Last layer layer of a neural network usually contains only one unit
- if activation function of last layer is :
	- linear : regression
	- logistic : binary classification
- Activation function should be differentiable for gradient descent to find optimal parameters w and b
- ACTIVATION FUNCTION : 
	- ReLU not differentiable at 0.
	- TanH : hyperbolic tangent function like logistic function but ranging from -1 to 1
	- ReLU : rectified linear unit = 0 when input is negative or z otherwise.
	- tanh(z) = (e^z - e^-z)/(e^z + e^-z)
	- ReLU(z) = 0 if z<0
			  = z if z>=0
- W is matrix , w is vector , b is vector
- W has rows of w and dimensions of w is the number of neurons in that layer
- Wz results a vector al= [wl,1z,wl,2z, . . . ,wl,sizelz]
- al + bl = cl
- g(cl) produces yl = [y1,y2,y3....] as output

## DEEP LEARNING

- training NN with more than 2 non - output layers

-BIG CHALLENGES : 
	- exploding gradient
		- solved - gradient clipping , L1, L2 regularization
	- vanishing gradient
		- BACKPROPOGATION :
			- To update the values of the parameters in NN
			- is an efficient algorithm for computing gradients on neural networks using the chain rule.
			- the neural network’s parameters receive an update proportional to the partial derivative of the cost function with respect to the current parameter in each iteration of training
			- in some cases, the gradient will be very small => preventing some parameters from changing their value => completely stop NN from training.
			- Eg : Tanh have gradient range (0,1) => gradient decreases for left layers making training slow for initial layers
		- solved :
			- ReLU
			- LSTM
			- technique like skip connection used in residual NN
			- advanced modification of Gradient Descent

## CONVOLUTIONAL NN

- In MLP, adding extra layers => adds more parameters => grows to big model => computationally intensive
- CNN => special kind of FFNN => reduces the number of parameters with many units without loosing quality of the model
- train different regression models on different pathes of an image to learn a different feature.
- patch => window size = p ; 3x3 
- sliding window makes patches
- each regression model is FFNN except layer 2 and 3.
- convolution = P X F where F is filter
- if convolution is high => similar F to P
- So, you can see the more the patch “looks” like the filter, the higher the value of the convolution operation is.  
- For convenience, there’s also a bias parameter b associated with each filter F which is added to the result of a convolution before applying the nonlinearity (activation function).
- One layer of a CNN consists of multiple convolution filters (each with its own bias parameter)
- Each filter of the first (leftmost) layer slides — or convolves — across the input image, left to right, top to bottom, and convolution is computed at each iteration
- FILTER matrix => trainable parameters => optimmized using gradient descent with backpropogation
- Typically, ReLU activation function is used in all hidden layers
- activation function of the output layer depends on the task
- If the CNN has one convolution layer following another convolution layer, then the subsequent layer l + 1 treats the output of the preceding layer l as a collection of sizel image matrices. Such a collection is called a volume. The size of that collection is called the volume’s depth. Each filter of layer l + 1 convolves the whole volume. The convolution of a patch of a volume
is simply the sum of convolutions of the corresponding patches of individual matrices the volume consists of.
- PROPERTIES:
	- STRIDE :
		- step size in moving window
		- high? => smaller output matrix smaller
	- PADDING :
		- allows getting a larger output matrix
		- it’s the width of the square of additional cells with which you surround the image (or volume) before you convolve it with the filter
- POOLING :
	- pooling layer applies a fixed operator, usually either max or average
	- hyperparameter: 
		- size of filter (2 or 3)
		- stride (2)
	- maxpooling > average pooling
	- increased accuracy
	- increases speed of training by reducing the number of parameters

## RNN

- used to label, classify or generate sequence.
- a sequence is a matrix => each row is a feature vector => order of row matters.
- To label a sequence is to predict a class for each feature vector in a sequence.
- To classify a sequence is to predict a class for the entire sequence.
- To generate a sequence is to output another sequence (of a possibly different length) somehow relevant to the input sequence.
- Used often in text processing - sentences -> sequence of words/characters. 
- also in speech processing.
- Not Feed-forward : contains loops
- State / Memory of unit : for each neuron/unit in a layer
- each unit u in each layer l receives two inputs: 
	- a vector of states from the previous layer l − 1 
	- the vector of states from this same layer l from the previous time step (memory).
- each training example is a matrix in which each row is a feature vector 
- Eg : X = [x1, x2, . . . , xt−1, xt, xt+1, . . . , xlengthX]
	- If our input example X is a text sentence, then feature vector xt for each t = 1, . . . , lengthX represents a word in the sentence at position t.
- an input example are “read” by the neural network sequentially in the order of the timesteps
- TO UPDATE STATE h(t): at each time t : linear combination of input vector and h(t-1)
	- The linear combination of two vectors is calculated using two parameter vectors wl,u, ul,u and a parameter bl,u
	- then apply activation function g to the linear combination to obtain h(t)
	- Typical ACTIVATION g : Tanh
- OUTPUT -> a vector calculated for the whole layer l at once
	- we use activation function g2 that takes a vector as input and returns a different vector of the same dimensionality 
	- The function g2 is applied to a linear combination of the state vector values htl,u calculated using a parameter matrix Vl and a parameter vector cl,u.
	- Typical ACTIVATION g2 : SOFTMAX FUNCTION
		- softmax function is a generalization of the sigmoid function to multidimensional outputs
	- Hyperparameter - dimentionality of V is chosen for multiplication with vector h so that results in vector in same dimensionality with c
	- Also depends on dimension of label
- Parameters : wl,u, ul,u, bl,u, Vl,u, and cl,u are computed from the training data using gradient descent with backpropagation 
- Special backpropogation : BACKPROPOGATION THROUGH TIME

- Problems: 
	- Tanh and softmax suffer vanishing gradient
	- handling long-term dependencies
		- the feature vectors from the beginning of the sequence tend to be “forgotten,” because the state of each unit, which serves as network’s memory
		- in text or speech processing, the cause-effect link between distant words in a long sentence can be lost
- GATED RNNs :
	- Long Short Term Memory : LSTM
	- Gated Recurrent Unit : GRU
	- Can store information for future use
	- reading, writing, and erasure of information stored in each unit is controlled by activation functions that take values in the range (0, 1)
	- The trained neural network can “read” the input sequence of feature vectors and decide at some early time step t to keep specific information about the feature vectors. That information about the earlier feature vectors can later be used by the model to process the feature vectors from near the end of the input sequence.
	- Units make decisions about what information to store, and when to allow reads, writes, and erasures. Those decisions are learned from data and implemented through the concept of gates.
	- minimal gated GRU :
		- g1 is the tanh activation function, g2 is called the gate function and is implemented as the sigmoid function taking values in the range (0, 1).
		- input: h(t-1) and xt
		- If gate is close to 0 => memory cells keep value from previous timestamp h(t-1).
		- If gate is close to 0 => the value of the memory cell is overwritten by a new value ht
		- g3 is SOFTMAX
	- when a network with gated units is trained with backpropagation through time, the gradient does not vanish
	- bi-directional RNNs
	- RNNs with attention
	- sequence-to-sequence RNN
	- a generalization of RNN is Recursive NN
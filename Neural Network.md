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
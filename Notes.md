# ML model goal - find optimal Parameters by tuning hyperparameters  

## Few definitions :

### Derivatives: 
	- of a function describes how fast the function grows or decreases.
	- processes of finding derivates is Deferrentiation.
### Gradient : 
	- generalization od derivative
	- a vector of partial derivatives by focusing on one of the function's input and considering all other inputs as constant
### Likelihood:
	- how likely an observation is according to our model.

## SVM  

- hyperplanes separating boundaries
- wx-b=0 -> divides the two linearly seperable clases
- margins --> wx-b=1 ; wx-b=-1
- distance = 2/w
- smaller w --> larger distance , bigger margin
- draws seperation as far as possible from the classes.
- linear separation --> linear kernel
- different separation --> poly etc kernels

- Sometimes difficult in perfectly seperating classes => OUTLIERS
- HYPERPARAMETER :
	- Penality for misclassification of training examples of specific classes 
	- weighting of each class - learning algorithm tries to not make errors in predicting training examples of this class - used when instances are minority in training but would like to avoid missclassfying
- OUTPUT:
	- a class

Two main problems:
- Dealing with noise:
	- no hyperplane can perfectly separate positive examples from negative ones
	- HINGE LOSS FUNCTION : max (0, 1 − yi(wxi − b)).
		- Hinge loss function = 0 if wxi lies on right side of decision boundary.
		- Hinge loss function is proportional to distance from decision boundary if wxi lies on wrong side of decision boundary
	- Minimize this loss function : C||w||** 2 + 1/N X SUM (max (0, 1 − yi(wxi − b)) )
		- C (hyperparameter) => tradeoff between increasing the size of the decision boundary and ensuring that each xi lies on the correct side of the decision boundary.
		- high C -> find highest margin -> ignores misclassification 
		- low C -> fewer misclassification by sacrificing margin size
	- SVM <--- HARD - MARGIN SVM
	- SVM + optimization of hinge loss <--- SOFT - MARGIN SVM
- Dealing with inherit non linearity:
	- cannot be separated by hyperplane
	- solution: transform original space into space of higher dimensionality - hopes to be linearly separable in this transformed space
	- this transformation - during cost function optimization -> KERNEL TRICK
	- Without transformation KERNELS -  
		- RBF kernel - | x-x'|^2/2rho^2
		- rho (hyperparameter) - decides smooth or curvy decision boundary


### SVM vs Linear Regression : 
Decision boundary is chosen in SVM as far as possible while separating two classes

|  +   /   -

|++   /    - - 

|+   /   - - -

|   /   - -   

|_______________



Hyperplane is chosen in Linear Regression as close as possible to all training examples.

|   ---/-

|  ---/- - 

|----/- - -

| --/- -   

|_______________



## Linear Regression

- Model f(x)=wx+b
- Goal : Best fitting Regression line/Hyperplane => prediction for new data has more chances to be correct 
- D-dimensional vecor = x ; y label
- OPTIMIZARION 
	- optimal w and b
	- minimize 1/N SUM(f(xi) − yi)square. <== LOSS FUNCTION , squared error loss (this type. Other types: absolute difference, binary loss, cube of difference)
	- minimize COST FUNCTION = minimize average LOSS FUNCTION or mean squared error or EMPERICAL RISK
- Rarely overfits.
- why sum of squares as loss? 
	- absolute values do not have continuous derivative => function is not smooth => difficult for algebraic operations i.e. gradient descent.
	- squared because 3 or 4 derivatives are complicated.
	- 

### Linear Regression vs Logistic Regression:
Linear Regression - minimize cost function
Logistic Regression - maximize the likelihood of training set according to model.

## Logistic Regression 

- classification
- binary / multiclass classification
- Sigmoid function / Standard logistic function : 1/1+e**-(wx+b)
- Model f(x)=1/1+e**-(wx+b)
- Hyperparamete: threshold
- OPTIMIZATION : 
	- Maximum likelihood
	- optimum w and b such that likelihood of output (predicted y) to be in a class is p ,0< p <1,is maximum
	- product: (f(xi) ** (yi))(1-f(xi))(1-yi) => for y=1 -> f(xi) ; y=0 -> 1-f(xi)
	- product of likelihoods of each observation
	- applying log => becomes log-likelihood
	- SUM (yi)log(f(x))+(1-yi)log(1-f(xi)) --------------------------------------------- (equi)
	- Log is strictly increasin function => maximizing function = maximizing arguments
	- Gradient descent
- OUTPUT:
	- score between 0 and 1
	- how confident the model is about the prediction or as the probability that the input example belongs to a certain class4

### Logistic Regression vs ID3:
Logistic Regression - builds parametric model fw,b by finding optimal solution 
ID3 - optimizes approximately by building non - parametric model fID3(x)= P(y=1|x) 

## Decision Tree Learning

- Acyclic graph leading to decision
- In each branch, a specific feature from feature vector is examined.
- If the value of feature is below a specific threshold => left branch is followed. else right.
- Leaf node => decision about the class is made
- labels belong to the set {0,1}
- ID3 algorithm
- OPTIMIZATION :
	- average log-likelihood
	- 1/N (equi)
	- every j feature of x vector is examined
	- for each j and threshold t, subsets are divided to S+ and S-. The best values (j,t) is picked and continue recursively in S+ and S-
	- stop when no split produces a model better than the current one.
	- To evaluate a good split - ENTROPY - measure of uncertainity about a random variable.
		- Maximum entropy -> when all random variables have equal probabilities.
		- ENTROPY: H(S) = -fID3 log fID3 - (1-fID3)log(1-fID3)
		- ENTROPY of a split H(S-,S+) = |S-|/|S|* H(S-) + |S+|/|S|* H(S+)
		- Therefore, at each step/leaf node, find the split that maximizes the entropy or stop.
		- stops at situation:
			- all examples are classified correctly
			- no attribute to split
			- split reduces entropy less than "epsilon" (hyperparameter)
			- tree reaches max depth (hyperparameter)
	- BACKTRACKING - since decision to split is local to iteration, backtracking during search for optimal decision tree -> longer to build

- Mostly widely decision tree learning algorithm - C4.5
- C4.5 vs ID3 : 
	- accepts both continuous and discrete features
	- handles incomplete examples
	- solves overfitting problem by using a bottom-up technique known as PRUNING.
		- PRUNING - going back up on tree once created, removing unwanted(no significant contribution to error reduction) branches . 
			- Eg: entropy = 0 (when all examples in S have the same label) => useless
			- Eg: entropy = 1 (when exactly one-half of examples in S is labeled with 1)
	- TO BE READ: how maximize avg log-likelihood
- OUTPUT:
	- score between 0 and 1
	- how confident the model is about the prediction or as the probability that the input example belongs to a certain class4


## KNN

- non-parametric learning method
- keeps the training data in memory
- new x comes in test, KNN find k training examples closest to x and returns majority label(classification) or avg label (regression).
- Distace metric (hyperparameter):
	- Euclidean	distance
	- Cosine similarity
		- measures similarity of direction of two vectors
		- cosine = 1 => same direction ; angle=0
		- cosine = 0 => angle = 90
		- cosine = -1 => opposite direction
		- cosine * -1 for distance metric
	- Chebychev distance
	- Mahalanobis distance
	- Hamming distance


# GRADIENT DESCENT

- iterative optimization algorithm for finding the minimum of a function.
- start at random point, take steps proportional to negative of gradient of function at the current point.
- Works for Linear Regression, Logistic Regression, SVM and Neural Network to find optimal parameters.
- Logistic and SVM optimization criterion -> convex -> only one minimum -> global
- NN -> not convex -> still good to find a local minimum.
Step 1 - partial derivatives of all parameters
step 2 - EPOCHs - one epoch consists of using the training set entirely to update each parameter
step 3 - Learning rate alpha - controls the size of an update.
  => w <- w - alpha * partial derivative of line wrt w
  => b <- b - alpha * partial derivative of line wrt b
Step 4 - continue updating until convergence ->values for w and b don’t change much after each epoch

### Minibatch stochastic gradient descent (minibatch SGD)

- speeds up the computation by approximating the gradient using smaller batches (subsets) of the training data

### Adagrad

- version of SGD which scales alpha for each parameter according to the history of gradients => alpha is reduced for very large gradients

### Momentum

- helps accelerate SGD by orienting the gradient descent in the relevant direction and reducing oscillations.

### RMSprop and Adam

- used in NN training

# GENERAL POINTS

- decision tree learning, logistic regression, or SVM build the model using the whole dataset at once. Naïve Bayes, multilayer perceptron, SGDClassifier/SGDRegressor, PassiveAggressiveClassifier/PassiveAggressiveRegressor in scikit-learn can be trained iteratively, one batch at a time. Once new training examples are available, you can update the model using only the new data.
- decision tree learning, SVM, and kNN can be used for both classification and regression.



## Kernel Regression

- Liner regression - but what if not straight line?
- For D-dimensions, wx + wx^2 + wx^3 + .... + wx^D +b
- with d>3 finding right polynomial is difficult 
- non-parametric method => no parameters to learn
- model based on data itself
- kernel plays imp role in similarity. 
- f(x)=1/N SUM(wiyi) for wi = Nk((xi-x)/b)/SUM(k(xi-x)/b)
- different forms. most used kernel : Gaussian kernel : k(z) = 1/sqrt(2pie) * e^(-z^2/2)
- b is hyperparameter-> tuned using validation set

## Multiclass classification

- logistic : from binary to multiclass by changing sigmoid to softmax
- SVM is binary. cannot be naturally extended to multiclass. 
	- => one vs rest

## One class classification

- unary classification or class modeling
- tries to identify objects of a specific class among all objects, by learning from a training set containing only the objects of that class
- Eg: outlier detection, anomaly detection, and novelty detection
- one-class Gaussian
	- we model our data as if it came from a Gaussian distribution, more precisely multivariate normal distribution (MND).
- one-class k-means
- one-class kNN
- one-class SVM

## Multilabel classification

- one vs rest
- algorithms that naturally can be made multiclass (decision trees, logistic regression and neural networks among others) can be applied to multi-label classification problems
- decide threshold
- NN trains multi-label cclassification models by using the binary cross-entropy cost function. 
	- The output layer of the neural network, in this case, has one unit per label
	- Each unit of the output layer has the sigmoid activation function

## Ensemble Learning

- for more complex algos => deep learning => requires more data. what if less data?
- boost performance of simple learning algorithms
- instead of trying to learn one super-accurate model, focuses on training a large number of low-accuracy models and then combining the predictions given by those weak models to obtain a high-accuracy meta-model 
- Low-accuracy models are usually learned by weak learners
- weak learner: decision tree => boost by combining large number of trees
- this combination is based on some sort of weighted voting.
- methods: 
	- boosting
		- by building each new model tries to “fix” the errors which previous models make
		- built iteratively
		- Eg: Gradient Boosting
	- bagging
		- creating many “copies” of the training data
		- then apply the weak learner to each copy to obtain multiple weak models and then combine them
		- Eg: Random forest

## Random Forest

- 
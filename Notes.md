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
	- Penality for misclassification of training examples of speific classes 

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
		- ENTROPY = -fID3 log fID3 - (1-fID3)log(1-fID3)

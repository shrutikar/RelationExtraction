# BASIC PRACTICE

## Cleaning

- shuffle
- ONE HOT ENCODING - catogorical data to vector
- BINNING/BUCKETING - converting numerical features into categorical based on a value range. => one hot encoding vector
- NORMALIZATION - an actual range of values which a numerical feature can take, into a standard range of values, typically in the interval [−1, 1] or [0, 1]
	- normalizing leads to increse speed of training
- STANDARDIZATION/z-score normalization - feature values are rescaled so that they have the properties of a standard normal distribution with μ = 0 and rho = 1,
	- = ˆx(j) = x(j) − μ(j)/pho(j)

- Unsupervised alg => more often benefit from standardization than from normalization  
- standardization as feature => if the values this feature takes are distributed close to a normal distribution 
- standardization as feature => if has extremely high or low values
- rest cases normalization is preferred

- MISSING VALUES:
	- if dataset big => remove that example
	- use DATA IMPUTATION TECHNIQUE
- DATA IMPUTATION TECHNIQUE :
	- replacing missing value in the feature with average value of the feature
	- replacing missing value in the feature with a value outside the normal range of values => learning alg will know to deal with values significantly different from normal range
	- replacing missing value in the feature with a vlue in the middle of the range => middle vlue will not affect the prediction significantly
	- Use missing value as target variables for a regression prob.
- Try several techniques, build several models and select the one that works the best

## Learning Algorithm Selection

- In-memory vs. out-of-memory:
	- large dataset? => incremental learning algorithms
- Number of features and examples
	- neural networks and gradient boosting => huge number of examples and features
	- SVM - limited
- Nonlinearity of the data
	- linearly separable? => SVM with linear kernel, logistic or linear regression
	- else deep NN or ensemble
- Training speed
	- NN slow to train
	- logistic and linear regression or decision trees faster
	- Random forest with multiple CPU cores
- Prediction speed
	- SVMs, linear and logistic regression, and (some types of) neural networks => fast
	- KNN, very deep or RNN => slow
- By testing on VALIDATION set
	- train - validation - test
	- 70 - 15 - 15
	- 95 - 2.5 - 2.5 <- big data
- decide by sklearn's ML algo selection diagram


## Underfitting vs Overfitting

- LOW BIAS / OVERFIT / HIGH VARIANCE - model predicts the training data well but fails on testing data
	- model is too complex for data (eg: tall decision tree or deep NN) => solution - try simpler model : linear instead of polynomial regression, or SVM with a linear kernel instead of RBF, a neural network with fewer layers/units
	- Too many features but small number of training examples => Reduce the dimensionality of the dataset
															  => Add more training data
															  => Regularize the model
- HIGH BIAS / UNDERFIT / LOW VARIANCE- model makes many mistakes on the training data
	- model is too simple for the data =>solution - try more complex model
	- features not informative =>solution - better features

## REGULARIZATION

- prevent overfitting
- methods that force the learning algorithm to build a less complex model
- leads to BIAS-VARIANCE TRADEOFF
- BIAS-VARIANCE TRADEOFF : leads to slightly higher bias but significantly reduces the variance
- Types: L1 and L2 regularization
- To create a regularized model - modify the objective function by adding a penalizing term whose value is higher when the model is more complex
- for linear regression : min[1/N * SUM (fw,b(xi) − yi)^2]
	- regularized : min[C|w| + 1/N * SUM(fw,b(xi) − yi)^2]
	- |w| and C are hyperparameters <- control the importance of regularization
	- C=0 -> original model
	- C=high -> learning alg tries to set |wj| to small value or 0 => model becomes simpler => underfit
	- As data analyst -> right C value to increase bias but decrease variance.
- L1 / LASSO REGULARIZATION:
	- produses a sparse model (has most of its parameters = 0) when C is large enough
	- therefore L1 does feature selection
	- increases model explainability
- L2 / RIDGE REGULARIZATION :
	- if your intention is not feature selection but just to maximize the model performance in hold-out set
	- differentiable i.e., Gradient Descent can be used for optimizing the objective function
- ELASTIC NET REGULARIZATION:
	- L1 and L2 combined
- DROPOUT :
	- NN
- BATCH NORMALIZATION :
	- NN
- DATA AUGUMENTATION: 
- EARLY STOPPING: 
- L1 and L2 used with linear as well as NN which directly minimize the objective function

## MODEL PERFORMANCE

- Regression:
	- predicted values close to the observed data values
	- Mean squared error (MSE) for training separately and testing separately.
		- if MSE_test > MSE_train => OVERFITTING => Regularize/Hyperparameter tune
- Classification:
	- 
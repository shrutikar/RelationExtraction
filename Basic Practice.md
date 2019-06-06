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
	- Each time you run a training example through the network, you temporarily exclude at random some units from the computation. The higher the percentage of units excluded the higher the regularization effect
	- add a dropout layer between two successive layers, or you can specify the dropout parameter [0,1] for the layer
- BATCH NORMALIZATION :
	- NN
	- standardizing the outputs of each layer before the units of the subsequent layer receive them as input
	- results in faster and more stable training, as well as some regularization effect
	- always insert a batch normalization layer between two layers
- DATA AUGUMENTATION: 
	- mostly with images
	- create synthetic dataset : zooming it slightly, rotating, flipping, darkening, and so on
- EARLY STOPPING: 
	- train a neural network by saving the preliminary model after every epoch and assessing the performance of the preliminary model on the validation set
	- at some point, after some epoch e, the model can start overfitting: the cost keeps decreasing, but the performance of the model on the validation data deteriorates
	- you can keep running the training process for a fixed number of epochs and then, in the end, you pick the best model. Models saved after each epoch are called checkpoints.
- L1 and L2 used with linear as well as NN which directly minimize the objective function

## MODEL PERFORMANCE

- Regression:
	- predicted values close to the observed data values
	- Mean squared error (MSE) for training separately and testing separately.
		- if MSE_test > MSE_train => OVERFITTING => Regularize/Hyperparameter tune
- Classification:
	- CONFUSION MATRIX:
   p1   p0
a1 TP   TN
a0 FP   FN
	- PRECISION/RECALL:
		- P = TP/TP+FP
		- R = TP/TP+FN
		- high P or high R by:
			- assigning a higher weighting to the examples of a specific class(SVM)
			- tuning hyperparameters to maximize precision or recall on the validation set
			- varying the decision threshold for algorithms that return probabilities of classes(Logistic regression)
	- ACCURACY:
		-TP + TN/(TP + TN + FP + FN)
		- number of correctly classified examples by total number of examples
	- COST SENSITIVE ACCURACY:
		- situation in which different classes have different importance
		- you first assign a cost (a positive number) to both types of mistakes: FP and FN
		- compute the counts TP, TN, FP, FN as usual and multiply the counts for FP and FN by the corresponding cost before calculating the accuracy 
	- AUC :
		- Area Under ROC Curve
		- ROC curve (receiver operating characteristic)
		- combination of the true positive rate (defined exactly as recall) and false positive rate (the proportion of negative examples predicted incorrectly)
		- TPR = TP/(TP + FN)
		- FPR = TP/(FP + TN)
		- ROC curves can only be used to assess classifiers that return some confidence score (or a probability) of prediction. For example, logistic regression, neural networks, and decision trees (and ensemble models based on decision trees)
		- First, discretize the range of the confidence score
		- a model is [0, 1], then you can discretize it like this: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
		- Then, you use each discrete value as the prediction threshold and predict the labels of examples in your dataset using the model and this threshold
		- The higher the area under the ROC curve (AUC), the better the classifier
		- A classifier with an AUC higher than 0.5 is better than a random classifier. If AUC is lower than 0.5, then something is wrong with your model

## HYPERPARAMETER TUNING

- GRID SEARCH :
	- when you have enough data to have a decent validation set, and number of hyperparameters and their range is not too large
	- Eg: SVM - C, kernerl
	- first time applying? -> don't know the range => use logarithmic scale. Eg : C -> [0.001, 0.01, 0.1, 1, 10, 100, 1000].
	- try each C with three kernels -> linear, RBF
	- then asses the performance
- RANDOM SEARCH :
	- you provide a statistical distribution for each hyperparameter from which values are randomly sampled and set the total number of combinations you want to try.
- BAYESIAN HYPERPARAMETER OPTIMIZATION :
	- uses past evaluation results to choose the next values to evaluate.
	- limit the number of expensive optimizations of the objective function by choosing the next hyperparameter values based on those that have done well in the past
- GRADIENT BASED TECHNIQUE
- EVOLUTIONARY OPTIMIZATION TECHNIQUES

## CROSS VALIDATION

- When you don’t have a decent validation set to tune your hyperparameters on
- you have few training examples => use more data to train the model
- split only training and testing and use cross-validation on training set
- First fix hyperparameter value
- Then you split your training set into several subsets of the same size => k fold
- use one set as validation and train on others, -> do for all set
- use metric on each and then average for final 
- Use grid search with cross-validation

## IMBALANCED DATASET

- SVM :
	- weights for missclassified (minority) classes
	- tries harder to correctly classify minority => but now missclassify majority class
- Oversampling : 
	- making multiple copies of the example from minority class
- Undersampling : 
	- remove examples from majority
- SMOTE (Synthetic minority oversampling technique) and ADASYN (Adaptice synthetic sampling method) : 
	- oversample synthetically => sampling feature values of several examples of the minority class and combining them to obtain a new example of that class
- Decision tree, RF, GB perform well on imbalanced datasets

## COMBINING MODELS
- averaging
- voting
- stacking : If some of your base models return not just a class, but also a score for each class, you can use these values as features too. 



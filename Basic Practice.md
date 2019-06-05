# BASIC PRACTICE

## Cleaning

- LOW BIAS - if model predicts the training data well
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


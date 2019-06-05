# BASIC PRACTICE

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

- MISSING VALUES - 
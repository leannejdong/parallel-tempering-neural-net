import numpy as np
X = np.random.randn(100,10)
mean_X = np.zeros((1,X.shape[1]))
std_X = np.ones((1,X.shape[1]))
for i in range(X.shape[1]):
	mean_X[:,i] = np.mean(X[:,i])
	std_X[:,i] = np.std(X[:,i])
	X[:,i] = (X[:,i] - mean_X[0,i])/std_X[0,i]
print(X)
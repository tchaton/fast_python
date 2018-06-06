import numpy as np
import time

numRows = 100000
numSamplesPerRow = 5000
x = np.random.rand(numRows, numSamplesPerRow)
limit = 100

def stats(tensor):
	return (np.size(tensor[0]), tensor[0].astype(np.float64), tensor[1].astype(np.float16))

def merge(left, right, axis=0):
	n_a, mean_a, var_a = stats(left)
	n_b, mean_b, var_b = stats(right)
	delta = mean_b - mean_a
	n_ab = n_a + n_b
	mean_ab = ((mean_a * n_a) + (mean_b * n_b)) / n_ab
	var_ab = (((n_a * var_a) + (n_b * var_b)) / n_ab) + ((n_a * n_b) * ((mean_b - mean_a) / n_ab)**2)
	#var_ab = var_a + var_b + ((delta*delta)/float(n_ab))*n_a*n_b
	return np.array([mean_ab, var_ab])



def mergeStd(tensor, axis=0, limit=100):
	if tensor.shape[0] < 100: ### limit can be adjusted. 100 was found to be better for this x
		return [np.mean(tensor, axis=axis), np.std(tensor, axis=axis)]

	middle = int(len(tensor)/2)
	left = mergeStd(tensor[:middle], axis=axis, limit=limit)
	right = mergeStd(tensor[middle:], axis=axis, limit=limit)
	
	return merge(left, right, axis=axis)

t0 = time.time()
[mean_ab, var_ab] = mergeStd(x, axis=0, limit=limit)
print(time.time() - t0)
t0 = time.time()
std = np.std(x, axis=0)
print(time.time() - t0)


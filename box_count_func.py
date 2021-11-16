import numpy as np
import matplotlib.pyplot as plt
from numba import jit

#---------------------------------------------------------------------#

# ---------Analysis of Multiscale Data from the Geosciences---------- #

# Author: Abhishek Harikrishnan
# Email: abhishek.harikrishnan@fu-berlin.de
# Last updated: 16-11-2021
# Update: Speed improvement with numba nopython mode
# Update: Testing with Sierpinski triangle

# REQUIREMENTS: NUMBA

# Source: F. Moisy 
# https://in.mathworks.com/matlabcentral/fileexchange/13063-boxcount

#---------------------------------------------------------------------#

@jit(nopython=True)
def computeFast(xlen, ylen, zlen, c, p, width, reverse_range, n, r):
	
	'''
	For a given number of resolutions r, the number of box counts n is
	computed.
	
	INPUT: 
	-> xlen, ylen, zlen: box dimensions
	-> c: 3D scalar field
	-> p: largest power of 2 to define box size such that r = 1, 2, .. , 2**p
	-> width: 2**p, size of the smallest cube which can embed the scalar field
	-> reverse_range: range of box sizes reversed in order
	-> n: empty array of size = size (reverse_range)
	-> r: box sizes in powers of 2 until 2**p
	
	OUTPUT:
	-> n: number of box counts for resolution r
	'''
	
	def getSum(xlen, ylen, zlen, _array):
		_sum = 0
		for i in range(xlen):
			for j in range(ylen):
				for k in range(zlen):
					if _array[i, j, k]:
						_sum = _sum + 1

		return _sum

	n[0] = getSum(xlen, ylen, zlen, c)

	for g in reverse_range:
		c_sum = 0
		siz = 2**(p-g)
		siz2 = int(round(siz/2))
		
		for i in range(0, int(width-siz)+1, int(siz)):
			for j in range(0, int(width-siz)+1, int(siz)):
				for k in range(0, int(width-siz)+1, int(siz)):
					c[i,j,k] = int(c[i,j,k]) or int(c[i+siz2, j, k]) or int(c[i, j+siz2, k]) or int(c[i+siz2, j+siz2, k]) or int(c[i,j,k+siz2]) or int(c[i+siz2,j,k+siz2]) or int(c[i,j+siz2,k+siz2]) or int(c[i+siz2,j+siz2,k+siz2])
					c_sum = c_sum + c[i,j,k]
		n[-g-1] = c_sum

	return n

def boxCountFunc(c, p, width):
	
	'''
	For a given 3D scalar field, the number of box counts n, the mean 
	fractal dimension meanDf and its standard deviation stdDf is computed.
	
	INPUT:
	-> c: 3D scalar field
	-> p: largest power of 2 to define box size such that r = 1, 2, .. , 2**p
	-> width: 2**p, size of the smallest cube which can embed the scalar field
	
	OUTPUT:
	-> n: number of box counts for resolution r
	-> r: box sizes in powers of 2 until 2**p
	-> meanDf: mean fractal dimension (computed only at the highest resolutions)
	-> stdDf: standard deviation (computed only at the highest resolutions)
	'''
	
	reverse_range = np.array([i for i in range(int(p))[::-1]])
	n = np.zeros((len(reverse_range) + 1), dtype = np.float32)
	r = 2**np.array(range(int(p)+1))
	
	xlen, ylen, zlen = np.shape(c)
	
	n = computeFast(xlen, ylen, zlen, c, p, width, reverse_range, n, r)
	
	return n, r


if __name__ == "__main__":
	
	print('Running test...')
	
	# Read the file quadraticKochCurve.png and compute the fractal dimension
	
	import imageio
	
	image = imageio.imread('Sierpinski.png')
	
	# Sum along the last dimension and reduce to 2d image
	# This step essentially adds all the RGBA components of the image
	
	image = np.sum(image, axis = -1)
	
	# Convert to boolean where all pixels having a value are set to true
	
	image = (image > 0)
	
	# Extend in 3D with 2D copies as boxCountFunc expects a 3D scalar field
	
	sizx, sizy = np.shape(image)
	
	numDim = 1
	tmp = np.zeros((sizx, sizy, numDim), dtype = np.bool)
	
	for i in range(numDim):
		tmp[:, :, i] += image
	
	image = tmp
	
	# Calculate largest power of 2 to define larger box size
	# The width is 2^p
	
	width = max(np.shape(image))
	p = np.log(width)/np.log(2)
	p = np.ceil(p)
	width = 2**p
	
	# Embed image in the larger box
	
	mz = np.zeros((int(width), int(width), int(width)), dtype = np.bool)
	mz[:np.shape(image)[0], :np.shape(image)[1], :np.shape(image)[2]] = image
	image = mz
	
	# Compute fractal dimension
	
	n, r = boxCountFunc(image, p, width)
	
	# Compute mean exponent
	# Depending on the box sizes used, one can choose which box sizes to compute the mean
	# For instance, the fractal dimension computed for the largest box size makes no sense
	# as the there will be only one point which has a value. 
	# One can always restrict the mean to the first few box sizes.
	
	df = - np.diff(np.log(n)) / np.diff(np.log(r))
	meanDf = np.mean(df[:-2])
	stdDf = np.std(df[:-2])
	
	print('Mean exponent is:', meanDf)
	print('Actual value is log_2(3): 1.585')
	print('Note: This value is obtained from an image. The accuracy highly depends on the quality of the image used.')

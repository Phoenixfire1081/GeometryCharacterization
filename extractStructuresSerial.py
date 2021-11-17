import numpy as np
import os
import time
import array
import copy
from struct import *
import time
from MCOutliers import outliers

#---------------------------------------------------------------------#

# ---------Analysis of Multiscale Data from the Geosciences---------- #

# Author: Abhishek Harikrishnan
# Email: abhishek.harikrishnan@fu-berlin.de
# Last updated: 17-11-2021
# Update: Testing with sample data

#---------------------------------------------------------------------#

def extractStructures(_threshValList, c, xlen, ylen, zlen, _zFastest, _writeNeighborInformation = False, _writePercolationData = False):

	# Timer

	start_time = time.time()

	for _threshVal in _threshValList:
		
		print('Processing threshold Value: ', _threshVal)
		
		if _zFastest:
			c = np.reshape(c, [xlen, ylen, zlen])
		
		else:
			c = np.reshape(c, [xlen, ylen, zlen], order = 'F')
		
		# Threshold the scalar field based on the sign of the value
		
		if _threshVal > 0:

			c = (c > _threshVal)
		
		else:
			c = (c < _threshVal)

		# Reshape arrays if grid directions are not of equal length

		mz = np.zeros((xlen + 1, ylen + 1, zlen + 1), dtype = np.bool)
		mz[:np.shape(c)[0], :np.shape(c)[1], :np.shape(c)[2]] = c
		c = mz

		#---------------------------------------------------------------------#

		# All boxes satisfying the thresholding criterion are checked
		# for their neighboring cells (26 - 6 faces, 12 edges, 8 corners). 

		_structVal = 0
		_structValuedGrid = np.zeros((xlen + 1, ylen + 1, zlen + 1), dtype = np.uint32)	# uint32 has values from 0 upto 4294967295
		_outlierCaseCount = 0

		for i in range(xlen + 1):
			for j in range(ylen + 1):
				for k in range(zlen + 1):
					
					if c[i,j,k]:
						
						if not _structValuedGrid[i,j,k] > 0:
							
								_structVal += 1
								
								print(_structVal)
								
								_neighborVal = []
								_structValuedGrid[i,j,k] = _structVal
								
								_loopCounter = 0
								_neighborCounter = 0
								
								_origiVal = i
								_origjVal = j
								_origkVal = k
								
								while True:
									
									# Reset all corner flags
								
									_c1 = False
									_c2 = False
									_c3 = False
									_c4 = False
									_c5 = False
									_c6 = False
									_c7 = False
									_c8 = False
								
									# Reset all face flags
								
									_f1 = False
									_f2 = False
									_f3 = False
									_f4 = False
									_f5 = False
									_f6 = False
									
									# Reset all edge flags
									
									_e1 = False
									_e2 = False
									_e3 = False
									_e4 = False
									_e5 = False
									_e6 = False
									_e7 = False
									_e8 = False
									_e9 = False
									_e10 = False
									_e11 = False
									_e12 = False
									
									_numberOfNeighbors = 0
									
									# Create aux cube [3, 3, 3]
								
									_i = 0
									_j = 0
									_k = 0
									_auxCube = np.zeros((3, 3, 3))
									
									if _loopCounter > 0:
										
										try:
										
											i = _neighborVal[_neighborCounter][0]
											j = _neighborVal[_neighborCounter][1]
											k = _neighborVal[_neighborCounter][2]
											
											_neighborCounter += 1
										
										except IndexError:
											
											break
									
									# Faces
									
									try:
										
										if c[i+1,j,k]:
											
											if not _structValuedGrid[i+1, j, k] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i+1,j,k] = _structVal
												_auxCube[_i+1,_j,_k] = 1
												_f1 = True
									
									except IndexError:
										
										pass
										
									try:
										
										if c[i-1,j,k]:
											
											if not _structValuedGrid[i-1, j, k] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i-1,j,k] = _structVal
												_auxCube[_i-1,_j,_k] = 1
												_f2 = True
												
									except IndexError:
										
										pass
									
									try:
										
										if c[i,j+1,k]:
											
											if not _structValuedGrid[i, j+1, k] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i,j+1,k] = _structVal
												_auxCube[_i,_j+1,_k] = 1
												_f3 = True
												
									except IndexError:
										
										pass
									
									try:
										
										if c[i,j-1,k]:
											
											if not _structValuedGrid[i, j-1, k] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i,j-1,k] = _structVal
												_auxCube[_i,_j-1,_k] = 1
												_f4 = True
												
									except IndexError:
										
										pass
									
									try:
										
										if c[i,j,k+1]:
											
											if not _structValuedGrid[i, j, k+1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i,j,k+1] = _structVal
												_auxCube[_i,_j,_k+1] = 1
												_f5 = True
												
									except IndexError:
										
										pass
										
									try:
										
										if c[i,j,k-1]:
											
											if not _structValuedGrid[i, j, k-1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i,j,k-1] = _structVal
												_auxCube[_i,_j,_k-1] = 1
												_f6 = True
									
									except IndexError:
										
										pass
										
									# Edges
									
									try:
										
										if c[i+1,j+1,k]:
											
											if not _structValuedGrid[i+1, j+1, k] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i+1,j+1,k] = _structVal
												_auxCube[_i+1,_j+1,_k] = 1
												_e1 = True
												
									except IndexError:
										
										pass
										
									try:
										
										if c[i+1,j-1,k]:
											
											if not _structValuedGrid[i+1, j-1, k] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i+1,j-1,k] = _structVal
												_auxCube[_i+1,_j-1,_k] = 1
												_e2 = True
									
									except IndexError:
										
										pass
									
									try:
										
										if c[i-1,j+1,k]:
											
											if not _structValuedGrid[i-1, j+1, k] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i-1,j+1,k] = _structVal
												_auxCube[_i-1,_j+1,_k] = 1
												_e3 = True
												
									except IndexError:
										
										pass
									
									try:
										
										if c[i-1,j-1,k]:
											
											if not _structValuedGrid[i-1, j-1, k] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i-1,j-1,k] = _structVal
												_auxCube[_i-1,_j-1,_k] = 1
												_e4 = True
									
									except IndexError:
										
										pass
									
									try:
										
										if c[i+1,j,k+1]:
											
											if not _structValuedGrid[i+1, j, k+1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i+1,j,k+1] = _structVal
												_auxCube[_i+1,_j,_k+1] = _structVal
												_e5 = True
									
									except IndexError:
										
										pass
										
									try:
										
										if c[i-1,j,k+1]:
											
											if not _structValuedGrid[i-1, j, k+1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i-1,j,k+1] = _structVal
												_auxCube[_i-1,_j,_k+1] = 1
												_e6 = True
												
									except IndexError:
										
										pass
									
									try:
										
										if c[i,j+1,k+1]:
											
											if not _structValuedGrid[i, j+1, k+1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i,j+1,k+1] = _structVal
												_auxCube[_i,_j+1,_k+1] = 1
												_e7 = True
									
									except IndexError:
										
										pass
										
									try:
										
										if c[i,j-1,k+1]:
											
											if not _structValuedGrid[i, j-1, k+1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i,j-1,k+1] = _structVal
												_auxCube[_i,_j-1,_k+1] = 1
												_e8 = True
												
									except IndexError:
										
										pass
										
									try:
										
										if c[i+1,j,k-1]:
											
											if not _structValuedGrid[i+1, j, k-1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i+1,j,k-1] = _structVal
												_auxCube[_i+1,_j,_k-1] = 1
												_e9 = True
												
									except IndexError:
										
										pass
									
									try:
										
										if c[i-1,j,k-1]:
											
											if not _structValuedGrid[i-1, j, k-1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i-1,j,k-1] = _structVal
												_auxCube[_i-1,_j,_k-1] = 1
												_e10 = True
												
									except IndexError:
										
										pass
									
									try:
										
										if c[i,j+1,k-1]:
											
											if not _structValuedGrid[i, j+1, k-1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i,j+1,k-1] = _structVal
												_auxCube[_i,_j+1,_k-1] = 1
												_e11 = True
									
									except IndexError:
										
										pass
										
									try:
										
										if c[i,j-1,k-1]:
											
											if not _structValuedGrid[i, j-1, k-1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i,j-1,k-1] = _structVal
												_auxCube[_i,_j-1,_k-1] = 1
												_e12 = True
												
									except IndexError:
										
										pass
										
									# Corners
									
									try:
										
										if c[i+1,j+1,k+1]:
											
											if not _structValuedGrid[i+1, j+1, k+1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i+1,j+1,k+1] = _structVal
												_auxCube[_i+1,_j+1,_k+1] = 1
												_c1 = True
												
									except IndexError:
										
										pass
										
									try:
										
										if c[i+1,j-1,k+1]:
											
											if not _structValuedGrid[i+1, j-1, k+1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i+1,j-1,k+1] = _structVal
												_auxCube[_i+1,_j-1,_k+1] = 1
												_c2 = True
												
									except IndexError:
										
										pass
										
									try:
										
										if c[i-1,j+1,k+1]:
											
											if not _structValuedGrid[i-1, j+1, k+1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i-1,j+1,k+1] = _structVal
												_auxCube[_i-1,_j+1,_k+1] = 1
												_c3 = True
												
									except IndexError:
										
										pass
										
									try:
										
										if c[i-1,j-1,k+1]:
											
											if not _structValuedGrid[i-1, j-1, k+1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i-1,j-1,k+1] = _structVal
												_auxCube[_i-1,_j-1,_k+1] = 1
												_c4 = True
												
									except IndexError:
										
										pass
										
									try:
										
										if c[i+1,j+1,k-1]:
											
											if not _structValuedGrid[i+1, j+1, k-1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i+1,j+1,k-1] = _structVal
												_auxCube[_i+1,_j+1,_k-1] = 1
												_c5 = True
												
									except IndexError:
										
										pass
										
									try:
										
										if c[i+1,j-1,k-1]:
											
											if not _structValuedGrid[i+1, j-1, k-1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i+1,j-1,k-1] = _structVal
												_auxCube[_i+1,_j-1,_k-1] = 1
												_c6 = True
												
									except IndexError:
										
										pass
										
									try:
										
										if c[i-1,j+1,k-1]:
											
											if not _structValuedGrid[i-1, j+1, k-1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i-1,j+1,k-1] = _structVal
												_auxCube[_i-1,_j+1,_k-1] = 1
												_c7 = True
												
									except IndexError:
										
										pass
										
									try:
										
										if c[i-1,j-1,k-1]:
											
											if not _structValuedGrid[i-1, j-1, k-1] > 0:
											
												_numberOfNeighbors += 1
												_structValuedGrid[i-1,j-1,k-1] = _structVal
												_auxCube[_i-1,_j-1,_k-1] = 1
												_c8 = True
												
									except IndexError:
										
										pass
									
									if _marchingCubesExt:
									
										# Marching cubes extension
										# All the points are checked again to see how the surface mesh is constructed
										
										_cornerFlags = [_c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8]
										_faceFlags = [_f1, _f2, _f3, _f4, _f5, _f6]
										_edgeFlags = [_e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8, _e9, _e10, _e11, _e12]
										
										_cube1 = [_f6, _e9, _c6, _e12, True, _f1, _e2, _f4]
										_cube2 = [_e10, _f6, _e12, _c8, _f2, True, _f4, _e4]
										_cube3 = [_c7, _e11, _f6, _e10, _e3, _f3, True, _f2]
										_cube4 = [_e11, _c5, _e9, _f6, _f3, _e1, _f1, True]
										_cube5 = [True, _f1, _e2, _f4, _f5, _e5, _c2, _e8]
										_cube6 = [_f2, True, _f4, _e4, _e6, _f5, _e8, _c4]
										_cube7 = [_e3, _f3, True, _f2, _c3, _e7, _f5, _e6]
										_cube8 = [_f3, _e1, _f1, True, _e7, _c1, _e5, _f5]
										
										# Identify the cases of the cube
										
										def find_case_number(_cube):
											
											return sum(2**v for v in range(8) if _cube[v] == True)
											
										_cases = [find_case_number(_cube1), find_case_number(_cube2), find_case_number(_cube3), find_case_number(_cube4), find_case_number(_cube5), find_case_number(_cube6), find_case_number(_cube7), find_case_number(_cube8)]
										
										sysErrFlag = False
										
										for ii in range(8):
											
											_caseVal = outliers(_cases[ii])
											
											if len(_caseVal) > 0:
												
												sysErrFlag = True
												_outlierCaseCount += 1
												
												# Keep the points which contain [i, j, k]. Discard all remaining ones.
												# NOTE: [i, j, k] differs for all cubes.
												
												if ii == 0: # corresponding to cube 1
													for jj in range(len(_caseVal)):
														if 4 not in _caseVal[jj]:
															_removeVal = _caseVal[jj]
															for kk in range(len(_removeVal)):
																if _removeVal[kk] == 0:
																	_structValuedGrid[i, j, k-1] = 0
																	_auxCube[_i, _j, _k-1] = 0
																if _removeVal[kk] == 1:
																	_structValuedGrid[i+1, j, k-1] = 0
																	_auxCube[_i+1, _j, _k-1] = 0
																if _removeVal[kk] == 2:
																	_structValuedGrid[i+1, j-1, k-1] = 0
																	_auxCube[_i+1, _j-1, _k-1] = 0
																if _removeVal[kk] == 3:
																	_structValuedGrid[i, j-1, k-1] = 0
																	_auxCube[_i, _j-1, _k-1] = 0
																if _removeVal[kk] == 5:
																	_structValuedGrid[i+1, j, k] = 0
																	_auxCube[_i+1, _j, _k] = 0
																if _removeVal[kk] == 6:
																	_structValuedGrid[i+1, j-1, k] = 0
																	_auxCube[_i+1, _j-1, _k] = 0
																if _removeVal[kk] == 7:
																	_structValuedGrid[i, j-1, k] = 0
																	_auxCube[_i, _j-1, _k] = 0
												
												if ii == 1: # corresponding to cube 2
													for jj in range(len(_caseVal)):
														if 5 not in _caseVal[jj]:
															_removeVal = _caseVal[jj]
															for kk in range(len(_removeVal)):
																if _removeVal[kk] == 0:
																	_structValuedGrid[i-1, j, k-1] = 0
																	_auxCube[_i-1, _j, _k-1] = 0
																if _removeVal[kk] == 1:
																	_structValuedGrid[i, j, k-1] = 0
																	_auxCube[_i, _j, _k-1] = 0
																if _removeVal[kk] == 2:
																	_structValuedGrid[i, j-1, k-1] = 0
																	_auxCube[_i, _j-1, _k-1] = 0
																if _removeVal[kk] == 3:
																	_structValuedGrid[i-1, j-1, k-1] = 0
																	_auxCube[_i-1, _j-1, _k-1] = 0
																if _removeVal[kk] == 4:
																	_structValuedGrid[i-1, j, k] = 0
																	_auxCube[_i-1, _j, _k] = 0
																if _removeVal[kk] == 6:
																	_structValuedGrid[i, j-1, k] = 0
																	_auxCube[_i, _j-1, _k] = 0
																if _removeVal[kk] == 7:
																	_structValuedGrid[i-1, j-1, k] = 0
																	_auxCube[_i-1, _j-1, _k] = 0
												
												if ii == 2: # corresponding to cube 3
													for jj in range(len(_caseVal)):
														if 6 not in _caseVal[jj]:
															_removeVal = _caseVal[jj]
															for kk in range(len(_removeVal)):
																if _removeVal[kk] == 0:
																	_structValuedGrid[i-1, j+1, k-1] = 0
																	_auxCube[_i-1, _j+1, _k-1] = 0
																if _removeVal[kk] == 1:
																	_structValuedGrid[i, j+1, k-1] = 0
																	_auxCube[_i, _j+1, _k-1] = 0
																if _removeVal[kk] == 2:
																	_structValuedGrid[i, j, k-1] = 0
																	_auxCube[_i, _j, _k-1] = 0
																if _removeVal[kk] == 3:
																	_structValuedGrid[i-1, j, k-1] = 0
																	_auxCube[_i-1, _j, _k-1] = 0
																if _removeVal[kk] == 4:
																	_structValuedGrid[i-1, j+1, k] = 0
																	_auxCube[_i-1, _j+1, _k] = 0
																if _removeVal[kk] == 5:
																	_structValuedGrid[i, j+1, k] = 0
																	_auxCube[_i, _j+1, _k] = 0
																if _removeVal[kk] == 7:
																	_structValuedGrid[i-1, j, k] = 0
																	_auxCube[_i-1, _j, _k] = 0
												
												if ii == 3: # corresponding to cube 4
													for jj in range(len(_caseVal)):
														if 7 not in _caseVal[jj]:
															_removeVal = _caseVal[jj]
															for kk in range(len(_removeVal)):
																	if _removeVal[kk] == 0:
																		_structValuedGrid[i, j+1, k-1] = 0
																		_auxCube[_i, _j+1, _k-1] = 0
																	if _removeVal[kk] == 1:
																		_structValuedGrid[i+1, j+1, k-1] = 0
																		_auxCube[_i+1, _j+1, _k-1] = 0
																	if _removeVal[kk] == 2:
																		_structValuedGrid[i+1, j, k-1] = 0
																		_auxCube[_i+1, _j, _k-1] = 0
																	if _removeVal[kk] == 3:
																		_structValuedGrid[i, j, k-1] = 0
																		_auxCube[_i, _j, _k-1] = 0
																	if _removeVal[kk] == 4:
																		_structValuedGrid[i, j+1, k] = 0
																		_auxCube[_i, _j+1, _k] = 0
																	if _removeVal[kk] == 5:
																		_structValuedGrid[i+1, j+1, k] = 0
																		_auxCube[_i+1, _j+1, _k] = 0
																	if _removeVal[kk] == 6:
																		_structValuedGrid[i+1, j, k] = 0
																		_auxCube[_i+1, _j, _k] = 0
												
												if ii == 4: # corresponding to cube 5
													for jj in range(len(_caseVal)):
														if 0 not in _caseVal[jj]:
															_removeVal = _caseVal[jj]
															for kk in range(len(_removeVal)):
																if _removeVal[kk] == 1:
																	_structValuedGrid[i+1, j, k] = 0
																	_auxCube[_i+1, _j, _k] = 0
																if _removeVal[kk] == 2:
																	_structValuedGrid[i+1, j-1, k] = 0
																	_auxCube[_i+1, _j-1, _k] = 0
																if _removeVal[kk] == 3:
																	_structValuedGrid[i, j-1, k] = 0
																	_auxCube[_i, _j-1, _k] = 0
																if _removeVal[kk] == 4:
																	_structValuedGrid[i, j, k+1] = 0
																	_auxCube[_i, _j, _k+1] = 0
																if _removeVal[kk] == 5:
																	_structValuedGrid[i+1, j, k+1] = 0
																	_auxCube[_i+1, _j, _k+1] = 0
																if _removeVal[kk] == 6:
																	_structValuedGrid[i+1, j-1, k+1] = 0
																	_auxCube[_i+1, _j-1, _k+1] = 0
																if _removeVal[kk] == 7:
																	_structValuedGrid[i, j-1, k+1] = 0
																	_auxCube[_i, _j-1, _k+1] = 0
												
												if ii == 5: # corresponding to cube 6
													for jj in range(len(_caseVal)):
														if 1 not in _caseVal[jj]:
															_removeVal = _caseVal[jj]
															for kk in range(len(_removeVal)):
																if _removeVal[kk] == 0:
																	_structValuedGrid[i-1, j, k] = 0
																	_auxCube[_i-1, _j, _k] = 0
																if _removeVal[kk] == 2:
																	_structValuedGrid[i, j-1, k] = 0
																	_auxCube[_i, _j-1, _k] = 0
																if _removeVal[kk] == 3:
																	_structValuedGrid[i-1, j-1, k] = 0
																	_auxCube[_i-1, _j-1, _k] = 0
																if _removeVal[kk] == 4:
																	_structValuedGrid[i-1, j, k+1] = 0
																	_auxCube[_i-1, _j, _k+1] = 0
																if _removeVal[kk] == 5:
																	_structValuedGrid[i, j, k+1] = 0
																	_auxCube[_i, _j, _k+1] = 0
																if _removeVal[kk] == 6:
																	_structValuedGrid[i, j-1, k+1] = 0
																	_auxCube[_i, _j-1, _k+1] = 0
																if _removeVal[kk] == 7:
																	_structValuedGrid[i-1, j-1, k+1] = 0
																	_auxCube[_i-1, _j-1, _k+1] = 0
												
												if ii == 6: # corresponding to cube 7
													for jj in range(len(_caseVal)):
														if 2 not in _caseVal[jj]:
															_removeVal = _caseVal[jj]
															for kk in range(len(_removeVal)):
																if _removeVal[kk] == 0:
																	_structValuedGrid[i-1, j+1, k] = 0
																	_auxCube[_i-1, _j+1, _k] = 0
																if _removeVal[kk] == 1:
																	_structValuedGrid[i, j+1, k] = 0
																	_auxCube[_i, _j+1, _k] = 0
																if _removeVal[kk] == 3:
																	_structValuedGrid[i-1, j, k] = 0
																	_auxCube[_i-1, _j, _k] = 0
																if _removeVal[kk] == 4:
																	_structValuedGrid[i-1, j+1, k+1] = 0
																	_auxCube[_i-1, _j+1, _k+1] = 0
																if _removeVal[kk] == 5:
																	_structValuedGrid[i, j+1, k+1] = 0
																	_auxCube[_i, _j+1, _k+1] = 0
																if _removeVal[kk] == 6:
																	_structValuedGrid[i, j, k+1] = 0
																	_auxCube[_i, _j, _k+1] = 0
																if _removeVal[kk] == 7:
																	_structValuedGrid[i-1, j, k+1] = 0
																	_auxCube[_i-1, _j, _k+1] = 0
												
												if ii == 7: # corresponding to cube 8
													for jj in range(len(_caseVal)):
														if 3 not in _caseVal[jj]:
															_removeVal = _caseVal[jj]
															for kk in range(len(_removeVal)):
																	if _removeVal[kk] == 0:
																		_structValuedGrid[i, j+1, k] = 0
																		_auxCube[_i, _j+1, _k] = 0
																	if _removeVal[kk] == 1:
																		_structValuedGrid[i+1, j+1, k] = 0
																		_auxCube[_i+1, _j+1, _k] = 0
																	if _removeVal[kk] == 2:
																		_structValuedGrid[i+1, j, k] = 0
																		_auxCube[_i+1, _j, _k] = 0
																	if _removeVal[kk] == 4:
																		_structValuedGrid[i, j+1, k+1] = 0
																		_auxCube[_i, _j+1, _k+1] = 0
																	if _removeVal[kk] == 5:
																		_structValuedGrid[i+1, j+1, k+1] = 0
																		_auxCube[_i+1, _j+1, _k+1] = 0
																	if _removeVal[kk] == 6:
																		_structValuedGrid[i+1, j, k+1] = 0
																		_auxCube[_i+1, _j, _k+1] = 0
																	if _removeVal[kk] == 7:
																		_structValuedGrid[i, j, k+1] = 0
																		_auxCube[_i, _j, _k+1] = 0
									
									if _auxCube[_i+1,_j,_k]:
										_neighborVal.append([i+1,j,k])
											
									if _auxCube[_i-1,_j,_k]:
										_neighborVal.append([i-1,j,k])
											
									if _auxCube[_i,_j+1,_k]:
										_neighborVal.append([i,j+1,k])
											
									if _auxCube[_i,_j-1,_k]:
										_neighborVal.append([i,j-1,k])
										
									if _auxCube[_i,_j,_k+1]:
										_neighborVal.append([i,j,k+1])
											
									if _auxCube[_i,_j,_k-1]:
										_neighborVal.append([i,j,k-1])
											
									if _auxCube[_i+1,_j+1,_k]:
										_neighborVal.append([i+1,j+1,k])
											
									if _auxCube[_i+1,_j-1,_k]:
										_neighborVal.append([i+1,j-1,k])
											
									if _auxCube[_i-1,_j+1,_k]:
										_neighborVal.append([i-1,j+1,k])
											
									if _auxCube[_i-1,_j-1,_k]:
										_neighborVal.append([i-1,j-1,k])
											
									if _auxCube[_i+1,_j,_k+1]:
										_neighborVal.append([i+1,j,k+1])
										
									if _auxCube[_i-1,_j,_k+1]:
										_neighborVal.append([i-1,j,k+1])
											
									if _auxCube[_i,_j+1,_k+1]:
										_neighborVal.append([i,j+1,k+1])
											
									if _auxCube[_i,_j-1,_k+1]:
										_neighborVal.append([i,j-1,k+1])
											
									if _auxCube[_i+1,_j,_k-1]:
										_neighborVal.append([i+1,j,k-1])
											
									if _auxCube[_i-1,_j,_k-1]:
										_neighborVal.append([i-1,j,k-1])
											
									if _auxCube[_i,_j+1,_k-1]:
										_neighborVal.append([i,j+1,k-1])
											
									if _auxCube[_i,_j-1,_k-1]:
										_neighborVal.append([i,j-1,k-1])
									
									if _auxCube[_i+1,_j+1,_k+1]:
										_neighborVal.append([i+1,j+1,k+1])
											
									if _auxCube[_i+1,_j-1,_k+1]:
										_neighborVal.append([i+1,j-1,k+1])
									
									if _auxCube[_i-1,_j+1,_k+1]:
										_neighborVal.append([i-1,j+1,k+1])
									
									if _auxCube[_i-1,_j-1,_k+1]:
										_neighborVal.append([i-1,j-1,k+1])
								
									if _auxCube[_i+1,_j+1,_k-1]:
										_neighborVal.append([i+1,j+1,k-1])
											
									if _auxCube[_i+1,_j-1,_k-1]:
										_neighborVal.append([i+1,j-1,k-1])
									
									if _auxCube[_i-1,_j+1,_k-1]:
										_neighborVal.append([i-1,j+1,k-1])
											
									if _auxCube[_i-1,_j-1,_k-1]:
										_neighborVal.append([i-1,j-1,k-1])
									
									_loopCounter += 1
										
								i = _origiVal
								j = _origjVal
								k = _origkVal

								if _writeNeighborInformation:
									
									_neighborVal.append([i, j, k])
									_neighborVal = np.array(_neighborVal)
									
									fw = open('NeighborInformation.txt', 'a')
									fw.write(str(_structVal) + ' ' + str(min(_neighborVal[:, 0])) + ' ' + str(max(_neighborVal[:, 0])) + ' ' + str(min(_neighborVal[:, 1])) + ' ' + str(max(_neighborVal[:, 1])) + ' ' + str(min(_neighborVal[:, 2])) + ' ' + str(max(_neighborVal[:, 2])) + '\n')
									fw.close()

		#---------------------------------------------------------------------#

		# Restore back to original grid

		_structValuedGrid = _structValuedGrid[:xlen, :ylen, :zlen]

		#---------------------------------------------------------------------#

		# Write out all data
									
		_structValuedGrid = _structValuedGrid.ravel()				

		u, counts = np.unique(_structValuedGrid, return_counts = True)

		print('Unique structure identifiers:', u)

		print('Counts for the unique structures:', counts)

		countsall = copy.deepcopy(counts)
		countsall.sort()
		print('Index of the biggest structure(s):', np.where(np.in1d(counts, countsall[::-1][1])))
		Vmax = countsall[::-1][1]
		print('Rearranged counts of structures:', countsall[::-1][:10])
		Vall = sum(countsall[::-1][1:])
		print('Sum of counts of all structures: ', Vall)

		if _writePercolationData:
			
			fw = open('Percolation_threshold.txt', 'a')
			fw.write(str(_threshVal) + ' ' + str(Vmax) + ' ' + str(Vall) + '\n')
			fw.close()

	print('Total time:', time.time() - start_time)
	
	return _structValuedGrid

if __name__ == "__main__":
	
	print('Running tests...')
	
	# Select file to run tests on
	_filenameRead = 'testData.bin'
	
	# What indexing does it use?
	# This refers to array order
	# See here for a full description: https://docs.oracle.com/cd/E19957-01/805-4940/z400091044d0/index.html
	# Python uses C-order indexing (or row-major order). For this set _zFastest = True
	_zFastest = True
	
	# Threshold values for testing
	_threshValList = [47]
	
	# Set additional parameters
	
	# Writes information relating to percolation analysis.
	_writePercolationData = False
	
	# Writes structure location information.
	# Useful for visualization purposes
	_writeNeighborInformation = True
	
	# Use Marching Cubes neighbor correction (more computation power required)
	_marchingCubesExt = True
	
	# Set data related parameters
	xlen = 200 
	ylen = 328
	zlen = 234
	
	# Set precision of binary data. 'f' is 32-bit floating point data.
	# For others, see here: https://docs.python.org/3/library/array.html
	precision = 'f'
	
	# Use array to read data. This is fast for binary files.
	data = array.array(precision)
	data.fromfile(open(_filenameRead, 'rb'), (xlen*ylen*zlen))
	
	# Convert to numpy array
	data = np.array(data, dtype = np.float32)
	
	# Initialize neighbor information file
	fw = open('NeighborInformation.txt', 'w')
	fw.close()
	
	# Find the structures at the threshold
	structureGrid = extractStructures(_threshValList, data, xlen, ylen, zlen, _zFastest, True)
	
	print('Expected number of structures at threshold 47:', 31)
	print('NOTE: structure 0 is empty space')

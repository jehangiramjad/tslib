######################################################
#
# Utility functions
#
######################################################
import numpy as np
from sklearn.metrics import mean_squared_error
import copy

def arrayToMatrix(npArray, nRows, nCols):

    if (type(npArray) != np.ndarray):
        raise Exception('npArray is required to be of type np.ndarray')

    if (nRows * nCols != len(npArray)):
        raise Exception('(nRows * nCols) must equal the length of npArray')

    return np.reshape(npArray, (nCols, nRows)).T


def matrixFromSVD(sk, Uk, Vk, probability=1.0):
    return (1.0/probability) * np.dot(Uk, np.dot(np.diag(sk), Vk.T))

def pInverseMatrixFromSVD(sk, Uk, Vk, probability=1.0):
    s = copy.deepcopy(sk)
    for i in range(0, len(s)):
        if (s[i] > 0.0):
            s[i] = 1.0/s[i]

    p = 1.0/probability
    return matrixFromSVD(s, Vk, Uk, probability=p)


def rmse(array1, array2):
    return np.sqrt(mean_squared_error(array1, array2))


def rmseMissingData(array1, array2):

    if (len(array1) != len(array2)):
        raise Exception('lengths of array1 and array2 must be the same.')

    subset1 = []
    subset2 = []
    for i in range(0, len(array1)):
        if np.isnan(array1[i]):
            subset1.append(array1[i])
            subset2.append(array2[i])

    return rmse(subset1, subset2)


def normalize(array, max, min):

    diff = 0.5*(min + max)
    div = 0.5 * (max - min)

    array = (array - diff)/div
    return array

def unnormalize(array, max, min):

    diff = 0.5*(min + max)
    div = 0.5*(max - min)

    array = (array *div) + diff
    return array


def randomlyHideValues(array, pObservation):

    count = 0
    for i in range(0, len(array)):
        if (np.random.uniform(0, 1) > pObservation):
            array[i] = np.nan
            count +=1 

    p_obs = float(count)/float(len(array))
    return (array, 1.0 - p_obs)


# following is taken from: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nanInterpolateHelper(array):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    (nans, x) = (np.isnan(array), lambda z: z.nonzero()[0])
    array[nans] = np.interp(x(nans), x(~nans), array[~nans])
    return array

#######################################################
# Testing 

# arr = [1,2.0,3.0,4,5,5,6,7,8,19, 29, 49]
# arr = np.array(arr)
# arr[0] = np.nan
# arr[8] = np.nan
# print(arr)
# arr = nanInterpolateHelper(arr)
# print(arr)


# N = 4
# T = 4
# data = np.random.normal(0.0, 10.0, N*T)
# #print(data)
# M = arrayToMatrix(data, N, T)

# # import algorithms.svdWrapper
# # from algorithms.svdWrapper import SVDWrapper as SVD
# # svdMod = SVD(M, method='numpy')
# # (sk, Uk, Vk) = svdMod.reconstructMatrix(4, returnMatrix=False)

# #M1 = matrixFromSVD(sk, Uk, Vk, probability=1.0)
# #print(np.mean(M - M1))

# (Uk, sk, Vk) = np.linalg.svd(M, full_matrices=False)
# Vk = Vk.T

# MA = np.linalg.pinv(M)
# MB = pInverseMatrixFromSVD(sk, Uk, Vk, probability=1.0)
# print(np.mean(MA - MB))

# M22 = matrixFromSVD(sk, Uk[0:-1, :], Vk, probability=1.0)

# M2 = pInverseMatrixFromSVD(sk, Uk[0:-1, :], Vk, probability=1.0)
# M4 = np.linalg.pinv(M)

# print(M2)
# print(M4)


# M3 = np.dot(np.dot(M, M2), M)
# print(np.mean(M - M3))
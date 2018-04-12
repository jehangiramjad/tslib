######################################################
#
# A wrapper for the SVD implementation of choice
#
######################################################
import numpy as np
from tslib.src.algorithms.pymf.svd import SVD as pSVD
from tslib.src import tsUtils

class SVDWrapper:

    def __init__(self, matrix, method='numpy'):
        if (type(matrix) != np.ndarray):
            raise Exception('SVDWrapper required matrix to be of type np.ndarray')

        self.methods = ['numpy', 'pymf']

        self.matrix = matrix
        self.U = None
        self.V = None
        self.s = None
        (self.N, self.M) = np.shape(matrix)

        if (method not in self.methods):
            print("The methods specified (%s) if not a valid option. Defaulting to numpy.linalg.svd" %method)
            self.method = 'numpy'

        else:
            self.method = method

    # perform the SVD decomposition
    # method will set the self.U and self.V singular vector matrices and the singular value array: self.s
    # U, s, V can then be access separately as attributed of the SVDWrapper class
    def decompose(self):

        # use the pymf library
        if (self.method == 'pymf'):
            pMod = pSVD(self.matrix)
            pMod.factorize()
            self.s = []
            for i in range(0, np.min(np.shape(pMod.S))):
                self.s.append(pMod.S[i, i])

            self.U = pMod.U[:, :]
            self.V = pMod.V[:, :]

        # default is numpy's linear algebra library
        else: 
            (self.U, self.s, self.V) = np.linalg.svd(self.matrix, full_matrices=False)

        # correct the dimensions of V
        self.V = self.V.T

    # get the top K singular values and corresponding singular vector matrices
    def decomposeTopK(self, k):

        # if k is 0 or less, just return empty arrays
        if (k < 1):
            return ([], [], [])

        # if k > the max possible singular values, set it to be that value
        elif (k > np.min([self.M, self.N])):
            k = np.min([self.M, self.N])

        if ((self.U is None) | (self.V is None) | (self.s is None)):
            self.decompose() # first perform the full decomposition

        print(self.s[0:10])
        sk = self.s[0:k]
        Uk = self.U[:, 0:k]
        Vk = self.V[:, 0:k]

        return (sk, Uk, Vk)

    # get the matrix reconstruction using top K singular values
    # if returnMatrix = True, then return the actual matrix, else return sk, Uk, Vk
    def reconstructMatrix(self, kSingularValues, returnMatrix=False):

        (sk, Uk, Vk) = self.decomposeTopK(kSingularValues)
        if (returnMatrix == True):
            return tsUtils.matrixFromSVD(sk, Uk, Vk)
        else:
            return (sk, Uk, Vk)



# ##################################################
# # Test code
# N = 30
# T = 50
# data = np.random.normal(0.0, 10.0, [N,T])

# mod1 = SVDWrapper(data, method='numpy')
# mod1.decompose()

# print(mod1.U)
# print("--")
# print(mod1.V)
# print("--")
# print(mod1.s)
# print("--")

# recon1 = mod1.reconstructMatrix(N)
# print(np.mean(data - recon1))
# recon1 = mod1.reconstructMatrix(int(N/2))
# print(np.mean(data - recon1))
# print("--")
# print("--")


# mod2 = SVDWrapper(data, method='pymf')
# mod2.decompose()

# print(mod2.U)
# print("--")
# print(mod2.V)
# print("--")
# print(mod2.s)
# print("--")

# recon2 = mod2.reconstructMatrix(N)
# print(np.mean(data - recon2))
# print("--")
# print("--")

# print(mod1.decomposeTopK(10))
# print(mod2.decomposeTopK(10))




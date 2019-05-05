import h5py
import numpy as np
import time
import tables
from math import ceil

def read_data(filename):
    return h5py.File(filename, "r+")


def write_data(filename, datalabel, matrix,mode = 'w'):
    f = h5py.File(filename, mode)
    f.create_dataset(datalabel, data=matrix)
    return True

def write_randomn_data(filename, matrixname, N, M, mean, sd):

    M = np.float64(np.random.normal(mean, sd, [N, M]))

    f = h5py.File(filename, "w")
    f.create_dataset(matrixname, data=M)

    f.close()
    return True

def write_randomn_data_seg(filename, matrixname, N, M, mean, sd, segment = None, max_memory = 10000*10000):
    if segment == None:
        segment = int(ceil(float(N)*M/max_memory))
    print ' writing data in ', segment, ' segments'
    dm = M/segment
    m1 = int(M - (segment-1)*dm)
    M = np.float64(np.random.normal(mean, sd, [N,m1 ]))

    f = h5py.File(filename, "w")
    f.create_dataset(matrixname, data=M,  maxshape=(None,None))
    for i in range(1,segment):
        M = np.float64(np.random.normal(mean, sd, [N, dm]))
        f[matrixname].resize(m1 + i * dm, axis=1)
        f[matrixname][-M.shape[0]:,-M.shape[1]:] = M
    f.close()
    return True

def copy_data(SourceFileName,dataName, filenameCopy):

    f = h5py.File(filenameCopy, "w")
    SourceFileName.copy(dataName,f)
    f.close()

    return

def copy_data_legacy(A, filenameCopy, matrixnameCopy):
    (n, m) = np.shape(A)
    f = h5py.File(filenameCopy, "w")
    f.create_dataset(matrixnameCopy, data=A)
    f.close()
    return
def transpose_data(A, filename, matrixname):
    (n, m) = np.shape(A)
    f = h5py.File(filename, "w")
    f.create_dataset(matrixname, data=A[:].T)
    f.close()


def add(A, B, Nchunk, subtract=False, overwriteA=False, matrixCFileName=None):

    atom = tables.Float64Atom()
    shape = (A.shape[0], A.shape[1])
    # check that dimensions are consistent
    assert A.shape[0] == B.shape[0]
    assert A.shape[1] == B.shape[1]

    # space for the new matrix, C
    if (overwriteA == False):
        filenameC = "00" + str(np.random.randint(0, Nchunk)) + "-addC.h5"
        if (matrixCFileName is not None):
            filenameC = matrixCFileName
        matrixCName = "C"

        h5f_c = tables.open_file(filenameC, 'w')

    # Nchunk = 1000  # ?
    chunkshape = (Nchunk, Nchunk)
    chunk_multiple = 1
    block_size = chunk_multiple * Nchunk

    if (overwriteA == False):
        C = h5f_c.create_carray(h5f_c.root, matrixCName, atom, shape, chunkshape=None)

    sz = block_size

    for i in range(0, A.shape[0], sz):
        for j in range(0, B.shape[1], sz):

            if (overwriteA == True):
                A[i:i + sz, j:j + sz] = np.add(A[i:i + sz, j:j + sz], B[i:i + sz, j:j + sz])
            else:
                C[i:i + sz, j:j + sz] = np.add(A[i:i + sz, j:j + sz], B[i:i + sz, j:j + sz])

    if (overwriteA == True):
        return A
    else:
        return C


def subtract(A, B, Nchunk, overwriteA=False, matrixCFileName=None):
    atom = tables.Float64Atom()
    shape = (A.shape[0], A.shape[1])
    # check that dimensions are consistent
    assert A.shape[0] == B.shape[0]
    assert A.shape[1] == B.shape[1]

    # space for the new matrix, C
    if (overwriteA == False):
        filenameC = "00" + str(np.random.randint(0, Nchunk)) + "-addC.h5"
        if (matrixCFileName is not None):
            filenameC = matrixCFileName
        matrixCName = "C"

        h5f_c = tables.open_file(filenameC, 'w')

    # Nchunk = 1000  # ?
    chunkshape = (Nchunk, Nchunk)
    chunk_multiple = 1
    block_size = chunk_multiple * Nchunk

    if (overwriteA == False):
        C = h5f_c.create_carray(h5f_c.root, matrixCName, atom, shape, chunkshape=None)

    sz = block_size

    for i in range(0, A.shape[0], sz):
        for j in range(0, B.shape[1], sz):
            if (overwriteA == True):
                A[i:i + sz, j:j + sz] = np.subtract(A[i:i + sz, j:j + sz], B[i:i + sz, j:j + sz])
            else:
                C[i:i + sz, j:j + sz] = np.subtract(A[i:i + sz, j:j + sz], B[i:i + sz, j:j + sz])

    if (overwriteA == True):
        return A
    else:
        return C

def dot(A, B,Nchunk, matrixCFileName=None, const=1.0, in_memory=False):
    # to write matrix C

    filenameC = "00" + str(np.random.randint(0, Nchunk*1000)) + "-multC.h5"
    matrixnameC = "C"

    if (matrixCFileName is not None):
        filenameC = matrixCFileName

    atom = tables.Float64Atom()
    shape = (A.shape[0], B.shape[1])

    # you can vary block_size and chunkshape independently, but I would
    # aim to have block_size an integer multiple of chunkshape
    # your mileage may vary and depends on the array size and how you'll
    # access it in the future.

    # space for the new matrix, C
    if (in_memory == False):
        h5f_c = tables.open_file(filenameC, 'w')

    # Nchunk = 1000  # ?
    chunkshape = (Nchunk, Nchunk)
    chunk_multiple = 1
    block_size = chunk_multiple * Nchunk

    if (in_memory == False):
        C = h5f_c.create_carray(h5f_c.root, matrixnameC , atom, shape, chunkshape=None)
    else:
        C = np.zeros([A.shape[0], B.shape[1]])

    sz = block_size

    for i in range(0, A.shape[0], sz):
        for j in range(0, B.shape[1], sz):
            for k in range(0, A.shape[1], sz):

                C[i:i + sz, j:j + sz] += np.dot(const * A[i:i + sz, k:k + sz], B[k:k + sz, j:j + sz])

    if (in_memory == True):
        return C[:]

    return (C, h5f_c)  # also return the fileObject so it can be closed

def dot2(A, B, Nchunk, matrixCFileName=None, const=1.0, in_memory=False,A_transpose = False, B_transpose = False ):
        # to write matrix C

        filenameC = "00" + str(np.random.randint(0, Nchunk * 1000)) + "-multC.h5"
        matrixnameC = "C"

        if (matrixCFileName is not None):
            filenameC = matrixCFileName

        atom = tables.Float64Atom()
        shape = [A.shape[0], B.shape[1]]
        if A_transpose: shape[0] = A.shape[1]
        if B_transpose: shape[1] = B.shape[0]
        # you can vary block_size and chunkshape independently, but I would
        # aim to have block_size an integer multiple of chunkshape
        # your mileage may vary and depends on the array size and how you'll
        # access it in the future.

        # space for the new matrix, C
        if (in_memory == False):
            h5f_c = tables.open_file(filenameC, 'w')

        # Nchunk = 1000  # ?
        chunkshape = (Nchunk, Nchunk)
        chunk_multiple = 1
        block_size = chunk_multiple * Nchunk

        if (in_memory == False):
            C = h5f_c.create_carray(h5f_c.root, matrixnameC, atom, tuple(shape), chunkshape=None)
        else:
            C = np.zeros([shape[0], shape[1]])

        sz = block_size
        if A_transpose and B_transpose:
            for i in range(0, A.shape[1], sz):
                for j in range(0, B.shape[0], sz):
                    for k in range(0, A.shape[0], sz):
                         C[i:i + sz, j:j + sz] += np.dot(const * A[k:k + sz, i:i + sz].T, B[j:j + sz,k:k + sz].T)
        elif B_transpose:
            for i in range(0, A.shape[0], sz):
                for j in range(0, B.shape[0], sz):
                    for k in range(0, A.shape[1], sz):
                        C[i:i + sz, j:j + sz] += np.dot(const * A[i:i + sz, k:k + sz], B[j:j + sz,k:k + sz].T)
        elif A_transpose:
            for i in range(0, A.shape[1], sz):
                for j in range(0, B.shape[1], sz):
                    for k in range(0, A.shape[0], sz):
                        C[i:i + sz, j:j + sz] += np.dot(const * A[k:k + sz, i:i + sz].T, B[k:k + sz, j:j + sz])
        else:
            for i in range(0, A.shape[0], sz):
                for j in range(0, B.shape[1], sz):
                    for k in range(0, A.shape[1], sz):

                        C[i:i + sz, j:j + sz] += np.dot(const * A[i:i + sz, k:k + sz], B[k:k + sz, j:j + sz])

        if (in_memory == True):
            return C[:]

        return (C, h5f_c)  # also return the fileObject so it can be closed

        # test

# N = 100
# M = 50
#
# mean = 10.0
# sd = 1.0
#
# filenameA = "dataA.hdf5"
# filenameB = "dataB.hdf5"
# filenameC = "dataC.h5"
# matrix_nameA = "A"
# matrix_nameB = "B"
# matrix_nameC = "C"
#
# # generate data
# write_randomn_data(filenameA, matrix_nameA, N, M, mean, sd)
# write_randomn_data(filenameB, matrix_nameB, N, M, mean, sd)
# write_randomn_data(filenameC, matrix_nameC, N, M, mean, sd)
#
# A = read_data(filenameA)[matrix_nameA]
# B = read_data(filenameB)[matrix_nameB]
# C = read_data(filenameC)[matrix_nameC]
# print ("data_generated")
# # numpy dot
#
# t1 = time.time()
# C1 = np.dot(A[:].T, B[:])
# #C1 = np.add(A, B)
# t2 = time.time()
#
# print t2 - t1
#
# # hdf5 mult
# t3 = time.time()
# (C2, fileobj) = dot2(A, B, Nchunk=100, A_transpose=True)
# #C2 = add(A, B,Nchunk=1000)
# t4 = time.time()
#
# print (t2 - t1, t4 - t3)
# print np.sum(C1-C2)
# # print np.mean(np.isclose(C1, C2))
# # print C1[0:10, 0:10]
# # print C2[0:10, 0:10]
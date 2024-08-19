import ctypes
import numpy as np
from ctypes import POINTER, c_int, c_size_t, Structure
from contextlib import contextmanager
import os

# Define the structure that matches the C++ struct
class PairVectors(Structure):
    _fields_ = [
        ("vec1", POINTER(c_int)),
        ("vec1_size", c_size_t),
        ("vec2", POINTER(c_int)),
        ("vec2_size", c_size_t)
    ]

# Load the shared library
mylib = ctypes.CDLL(f'{os.path.dirname(os.path.realpath(__file__))}/find_idx.so')

# Define the argument types and return type of the wrapper function
mylib.GetIndexes.argtypes = [
    POINTER(c_int), c_size_t, c_int, c_int
]
mylib.GetIndexes.restype = PairVectors

# Define the argument types for the free function
mylib.freeVecPair.argtypes = [PairVectors]
mylib.freeVecPair.restype = None

def fi_free(pv):
    mylib.freeVecPair(pv)

def find_indxs(vec, n, k):
    # Call the function
    vec = vec.astype(np.int32)
    result = mylib.GetIndexes(
        vec.ctypes.data_as(POINTER(c_int)),
        vec.size,
        n,
        k
    )

    # Convert the results to NumPy arrays
    vec1 = np.ctypeslib.as_array(result.vec1, shape=(result.vec1_size,))
    vec1 = np.copy(vec1)
    vec2 = np.ctypeslib.as_array(result.vec2, shape=(result.vec2_size,))
    vec2 = np.copy(vec2)
    fi_free(result)

    return (vec1, vec2)
    
if __name__ == "__main__":
    vec = np.array([[0,1,4],[2,1,3],[0,1,2]])
    vec1, vec2 = find_indxs(vec, 5, 3)
    print(vec)
    print(vec1)
    print(vec2)




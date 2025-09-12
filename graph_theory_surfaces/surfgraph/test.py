from numpy.linalg import norm
import numpy as np

# eps is expected error on floating point math
def normalize(vector, eps=1e-4):
    return vector / norm(vector) if norm(vector) > eps else vector * 0

# start from unnormalized vectors
vects = [[1, 0, 0],
         [1, 1, 1],
         [0, 1, 0],
         [0, 0, 1],
         [-1, 0, 0],
         [0, -1, 0]]

# normalize all vectors
vects = np.array([normalize(vect) for vect in vects])

# sum vectors, normalize, and invert
opposing = -normalize(vects.sum(axis=0))

# expect [0, 0, -1] for this example
print(opposing) 

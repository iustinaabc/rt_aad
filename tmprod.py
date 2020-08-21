import numpy as np

def tmprod(tensor,matrix):
    """
    Returns the product of a 3-dimensional tensor with a 2-dimensional matrix
    """
    # matrix * 3rd order tensor = 3rd order tensor
    prod = np.zeros((matrix.shape[0],tensor.shape[1],tensor.shape[2]),dtype=np.float32)
    for ind in range(tensor.shape[2]):
        prod[:,:,ind]= np.matmul(matrix[:,:],tensor[:,:,ind])

    return prod
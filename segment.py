import numpy as np
import math


def segment(vector,size):
    """
    Segments in the time dimension (DIMENSION 3RD) added to the last dimension (DIMENSION 4TH)
    """
    top = math.floor(vector.shape[2]/size)
    new_vector = np.full((vector.shape[0],vector.shape[1],size,vector.shape[3], top),0,dtype=np.float32)

    for x in range(0,top):
        new_vector[:,:,:size,:,x] = vector[:,:,x*size:(x+1)*size,:]

    new_vector = np.reshape(new_vector,(new_vector.shape[0],new_vector.shape[1],
                                        size,new_vector.shape[3]*new_vector.shape[4]))

    return new_vector

# a = np.array([    [   [ [1,2 ],[3,4] ],[[5,6],[7,8]],[[9,10],[11,12]] ,
#                       [[9,10],[11,12]], [[9,10],[11,12]],[[9,10],[11,12]] ],
#                   [   [ [1,2 ],[3,4]],[[5,6],[7,8]], [[9,10],[11,12]],
#                       [[9,10],[11,12]],[[9,10],[11,12]],[[9,10],[11,12]]  ]
#                   ])
#
# a = np.zeros((2,2,14,2))
# print(a.shape)
# print(segment(a,4).shape)
# print(60*64)
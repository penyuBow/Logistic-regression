import numpy as np 

def map_feature(x1, x2):
#   Returns a new feature array with more features, comprising of 
#   x1, x2, x1.^2, x2.^2, x1*x2, x1*x2.^2, etc.

    degree = 6
    out = np.ones([len(x1),((degree + 1) * (degree + 2) // 2)])
    idx = 1

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            a1 = x1 ** (i - j)
            a2 = x2 ** j
            out[:, idx] = a1 * a2
            idx += 1

    return out
    
"""x1 = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
x2 = np.array([10,11,12,13,14,15,16,17,18,19,20]).reshape(-1,1)
print(map_feature(x1,x2))
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
print(X)"""
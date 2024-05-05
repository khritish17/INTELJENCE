import numpy as np
a = np.zeros((1, 4))
a[0][0], a[0][1], a[0][2], a[0][3] = 1.47812643e-320,0.1,0.3,0.7 
print(a)
print(np.log(a))
b = np.clip(a, 1e-10, 1 - 1e-10)
print(b)
print(np.log(b))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
import os
import pickle,glob

from sklearn.datasets import make_regression
# X为样本特征，y为样本输出， coef 为回归系数，共1000个样本，每个样本1个特征
X, y = make_regression(n_samples=500, n_features=2, coef= False)




# with open('E:\dataset\hexBin.bin', 'rb') as fp:
#     for line in fp.readlines():
#         print(line)

# with open('E:\dataset\hexBin.bin','wb') as fp:
#     for x in arr:
#         a = struct.pack('f',x)
#         fp.write(a)





# df = pd.DataFrame(X,y)
# data = np.array(df)
# outfile = r'E:\dataset\test.npy'
# np.save(outfile,data)

# print("系数：",coef)
# print("返回的y值：",y)
# print("运算的y值",np.dot(X,coef.T))
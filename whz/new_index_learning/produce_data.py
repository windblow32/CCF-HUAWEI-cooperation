import numpy as np
from numpy import random
import pandas as pd
import csv
import matplotlib.pyplot as plt
import struct
import os
import pickle, glob


# from sklearn.datasets import make_regression
# # X为样本特征，y为样本输出， coef 为回归系数，共1000个样本，每个样本1个特征
# X, y = make_regression(n_samples=500, n_features=2, coef= False)


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


class TestData:

    def __init__(self, step, file_num):
        # step表示每个文件数量大小的增长
        # file_num表示生成的文件数量
        self.step = step
        self.file_num = file_num

    @staticmethod
    def create(self):
        for i in range(self.file_num):

            filename = 'data' + str(i) + '.csv'
            f = open(filename, 'w', encoding='utf-8')
            csv_writer = csv.writer(f)
            csv_writer.writerow(['x', 'y'])
            scale = (i + 1) * 100000
            x = random.normal(loc=(i + 1) * 100, size=scale)
            x = np.sort(x).reshape(-1)
            y = range(scale)
            for j in range(scale):
                csv_writer.writerow([str(x[j]), str(y[j])])
            f.close()

    @staticmethod
    # 读取某个文件，index表示该文件对应的下标
    def read(self, index):
        if index >= self.file_num or index < 0:
            print("输入的文件下标有误")
            return -1
        else:
            file_name = 'data' + str(index) + '.csv'
            df = pd.read_csv(file_name, sep=',')
            return df


if __name__ == "__main__":
    data = TestData(1000, 1)
    data.create(data)
    csv_data = data.read(data, 0)
    x_df = csv_data.iloc[:, 0]
    y_df = csv_data.iloc[:, 1]
    print(x_df)
    print(y_df)

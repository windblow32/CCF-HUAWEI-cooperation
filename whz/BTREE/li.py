import time
import numpy as np
from memory_profiler import profile
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import math


class TopModel:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def ttrain(self):
        return np.poly1d(np.polyfit(self.x, self.y, 3))


class LearnedIndex:

    def __init__(self, model_num):
        self.RMI = [1, model_num]  # Learned Index RMI架构
        self.index = []  # 存储模型的索引
        self.N = None  # key值总数
        self.pos = None
        self.data = None  # 所有key值
        self.error_bound = []  # 存储最后一层每个Model的 min_err and max_err
        self.average_error = 0
        self.mean = None  # 存储均值，数据标准化
        self.std = None  # 存储标准差，数据标准化
        self.build()

    def build(self):
        for m in self.RMI:
            if m == 1:
                model = TopModel(self.data, self.pos)
                self.index.append(model)
            else:
                # 第二層 => 建置多個簡單線性回歸
                self.index.append([])
                for j in range(m):
                    model = LinearRegression()
                    self.index[1].append(model)

    def train(self, data, pos):
        self.pos = pos
        self.data = data
        self.N = data.size
        y = self.crtCDF(data)
        norm_data = preprocessing.scale(data)  # 標準化: 零均值化
        self.mean = data.mean()
        self.std = data.std()
        # print("scale:",norm_data)
        # print("mine:",(data - self.mean)/data.std())

        for m in self.RMI:
            if m == 1:
                # 顶层采用三次多项式拟合
                self.index[0] = np.poly1d(np.polyfit(self.data, self.pos, 3))
            else:
                sub_data = [[] for i in range(m)]
                sub_y = [[] for i in range(m)]

                for i in range(self.N):
                    mm = int(self.index[0]([[norm_data[i]]]) * m / self.N)

                    if mm < 0:
                        mm = 0
                    elif mm > m - 1:
                        mm = m - 1

                    sub_data[mm].append(data[i])
                    sub_y[mm].append(y[i])

                # 训练第二层所有的 SLR Model
                for j in range(m):
                    xx = np.array(sub_data[j])
                    yy = np.array(sub_y[j])

                    min_err = max_err = 0
                    if xx.size > 0:
                        xx = np.reshape(xx, (-1, 1))
                        self.index[1][j].fit(xx, yy)

                        # 计算最后一层 Model 的 min_err/max_err
                        for i in range(data.size):
                            pred_pos, _ = self.predict(data[i])
                            err = abs(pred_pos - i)
                            # print(f"error: {err}")
                            self.average_error += err
                            if err < min_err:
                                min_err = math.floor(err)
                            elif err > max_err:
                                max_err = math.ceil(err)
                        self.average_error /= data.size
                        print(f"average error:{self.average_error / self.N * 100}%")
                    self.error_bound.append([min_err, max_err])

    def predict(self, key):
        mm = int(self.index[0]([[(key - self.mean) / self.std]]) * self.RMI[1] / self.N)
        if mm < 0:
            mm = 0
        elif mm > self.RMI[1] - 1:
            mm = self.RMI[1] - 1
        pred_pos = int(self.index[1][mm].predict([[key]]))
        return pred_pos, mm

    @profile
    def search(self, key):  # model biased search
        start = time.time()
        pos, model = self.predict(key)

        left = pos + self.error_bound[model][0]
        right = pos + self.error_bound[model][1]

        # 检测预测位置
        if pos < 0:
            left = pos = 0
        if pos > self.N - 1:
            right = pos = self.N - 1

        if left < 0:
            left = 0
        if right > self.N - 1:
            right = self.N - 1

        # print(left,pos,right)

        while left <= right:

            if self.data[pos] == key:
                end = time.time()
                print(f"Search Time: {end - start}")
                return True
            elif self.data[pos] > key:
                right = pos - 1
            elif self.data[pos] < key:
                left = pos + 1
            pos = int((left + right) / 2)
        end = time.time()
        print(f"Search Time: {end - start}")
        return False

    def crtCDF(self, x):
        if (type(x) == np.ndarray):
            loc = x.mean()
            scale = x.std()
            N = x.size
            pos = norm.cdf(x, loc, scale) * N
            return pos
        else:
            print("Wrong Type! x must be np.ndarray ~")
            return

    def calculate_wrong(self):
        return self.error_bound


def main():
    data = np.hstack((np.random.randint(1500000, size=1500000), np.random.normal(150000, 10, size=1500000),
                      np.random.uniform(1, 1500000, size=1500000), np.random.poisson(1500000, size=1500000),
                      np.random.exponential(15000000, size=1500000)))
    # data = np.random.randint(1, 10000, size=100)
    data = np.sort(data).reshape(-1)
    file = np.savetxt("E:\whz\LABDATA\multiData160M.csv", data, delimiter=",")
    li = LearnedIndex(2)
    start = time.time()
    li.train(data, np.arange(data.size))
    end = time.time()
    print(f'time:{end - start}s')
    # print(li.error_bound)
    # for k in data:
    #     print(li.search(k))
    li.search(data[100])


if __name__ == "__main__":
    main()

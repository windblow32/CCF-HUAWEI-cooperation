import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import random
from knn_class import KNN
from decision_tree_class import decision_tree
from rf_class import rf
from muti_class import muti_reg
from elastic_class import elastic_reg
from rbf_class import rbf
from GradientBoostingRegressor_class import gbr
from extraforest_class import etr
from linear_class import linear

# from sklearn.datasets import make_regression
# # X为样本特征，y为样本输出， coef 为回归系数，共1000个样本，每个样本1个特征
# X, y = make_regression(n_samples=500, n_features=2, coef= False)

# data = pd.read_csv('E:\whz\LABDATA\\normal_s.csv', header=None, engine='python')
# X = []
# y = []
# global TOTAL_NUMBER
# TOTAL_NUMBER = data.shape[0]
# print(TOTAL_NUMBER)
# for i in range(data.shape[0]):
#     X.append(data.loc[i, 0])
#     y.append(data.loc[i, 1])

# 导入IRIS数据集
iris = load_iris()
# 特征矩阵
X = iris.data
# 目标向量
y = iris.target


np.array(X)

# X.shape = (TOTAL_NUMBER,1)

# load 后加一列-1,作为初始化
# np.insert(X, 0, -1, axis=1)
# z存储所有的被选择的model序号
z = []
# size为z的大小
size = 0
print(y)
j = 0
temp = 0
read_line = 100
# 文件输出
# f = open('log.txt','w')
# for i in range(100):
#     print(str(i), file=f)
# f.close()
res = ['', '', '', '', '', '', '', '', '']
for seed in range(10, 50, 5):
    if temp + read_line <= X.shape[0]:
        # x = X[temp:temp + read_line, 1:]
        x = X[temp:temp + read_line]
        y1 = y[temp:temp + read_line]
        temp = temp + read_line
    else:
        # x = X[temp:, 1:]
        # y1 = y[temp:]
        # temp = X.shape[0]
        break;


    # x = X[:, 1:]
    # y1 = y
    print("***************")
    print("epoch " + str(j))
    j += 1
    # Median_AE 无关
    score1 = KNN(x, y1, seed).getscore()
    print(score1)

    score2 = decision_tree(x, y1, seed).getscore()
    print(score2)

    score3 = rf(x, y1, seed).getscore()
    print(score3)

    score4 = muti_reg(x, y1, seed).getscore()
    print(score4)

    score5 = elastic_reg(x, y1, seed).getscore()
    print(score5)

    score6 = rbf(x, y1, seed).getscore()
    print(score6)

    score7 = gbr(x, y1, seed).getscore()
    print(score7)

    score8 = etr(x, y1, seed).getscore()
    print(score8)

    score9 = linear(x, y1, seed).getscore()
    print(score9)

    score = [score1, score2, score3, score4, score5, score6, score7, score8, score9]
    name = ["KNN Regression", "Decision Tree Regression", "Random Forest Regression", "Multi Regression",
            "Elastic Regression", "Radial Basis Function Regression", "Gradient Boosting Regressor",
            "ExtraTree Regression", "Linear Regression"]

    mininum = min(score)
    min_index = score.index(mininum)
    mark = 0
    markedIndex = [-1 for x in range(0, 9)]
    for k in score:
        if (k - mininum) <= 0.2:
            index = score.index(k)
            markedIndex[mark] = int(index)
            res[int(index)] = name[index]
            mark += 1
    for i in range(temp-read_line, temp):
        marked = int(random.randint(0, mark - 1))
        X[i][0] = markedIndex[int(marked)]
        z.append(markedIndex[int(marked)])

    size += read_line;
    k = 0  # 第k个较为接近

    print("最优模型为：")
    print(name[min_index])
    print("一共推荐以下模型")
    t = 0
    for t in range(9):
        if (res[t] != ''):
            print(res[t], end='; ')
    print('')

print()

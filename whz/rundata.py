from sklearn.datasets import load_iris
import pandas as pd
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

# data = pd.read_csv('E:/whz/LABDATA/normal_s.csv', header=None, engine='python')
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
i = 0

# 文件输出
# f = open('log.txt','w')
# for i in range(100):
#     print(str(i), file=f)
# f.close()
res = ['','','','','','','','','']
for seed in range(10,50,5):
    print("***************")
    print("epoch " + str(i))
    i += 1
    score1 = KNN(X, y, seed).getscore()

    score2 = decision_tree(X, y, seed).getscore()

    score3 = rf(X, y, seed).getscore()


    score4 = muti_reg(X, y, seed).getscore()


    score5 = elastic_reg(X, y, seed).getscore()


    score6 = rbf(X, y, seed).getscore()


    score7 = gbr(X, y, seed).getscore()


    score8 = etr(X, y, seed).getscore()


    score9 = linear(X, y, seed).getscore()

    score = [score1, score2, score3, score4, score5, score6, score7, score8, score9]
    name = ["KNN Regression", "Decision Tree Regression", "Random Forest Regression", "Multi Regression", "Elastic Regression", "Radial Basis Function Regression", "Gradient Boosting Regressor", "ExtraTree Regression", "Linear Regression" ]


    mininum = min(score)
    min_index = score.index(mininum)
    k = 0 # 第k个较为接近
    for k in score:
        if(k-mininum) <= 3:
            index = score.index(k)
            res[int(index)] = name[index]


    # 数组下标加一是模型序号
    # print(name[min_index])
    # t = 0
    # for t in range(9):
    #     if(res[t] != ''):
    #         print(res[t], end='; ')
    # print('')

print(name[min_index])
t = 0
for t in range(9):
    if(res[t] != ''):
        print(res[t], end='; ')
print('')

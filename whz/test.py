import io
from importlib import reload

import sys

from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics

# 交叉验证
from sklearn.model_selection import StratifiedKFold
# 网格搜索
from sklearn.model_selection import GridSearchCV

# 指标
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 导入IRIS数据集
iris = load_iris()
# 特征矩阵
X = iris.data
# 目标向量
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state =1)

# 调参
# n_estimators=100, *, criterion='mse',
# max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
# max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
# verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None

regressor = RandomForestRegressor()

#kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)  # 将训练/测试数据集划分10个互斥子集，

grid_search = GridSearchCV(regressor,param_grid={'max_depth': range(20,50,5),'n_estimators':range(500,1000,100) },verbose=4,n_jobs=4,scoring='neg_mean_squared_error',cv = 10)
# scoring指定损失函数类型，n_jobs指定全部cpu跑，cv指定交叉验证
# 运行网格搜索


grid_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))
# rid_scores_：给出不同参数情况下的评价结果。best_params_：描述了已取得最佳结果的参数的组合
# best_score_：成员提供优化过程期间观察到的最好的评分
# 具有键作为列标题和值作为列的dict，可以导入到DataFrame中。
# 注意，“params”键用于存储所有参数候选项的参数设置列表。
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean,param in zip(means, params):
    print("%f  with:   %r" % (mean, param))

y_pred = regressor.predict(X_test)

print("预测结果：\n", y_pred)
print("真实值：\n", y_test)
# 计算RMSE(均方根误差，标准误差)
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("R2_score", r2_score(y_test,y_pred))

# median_absolute_error
print(median_absolute_error(y_test,y_pred))

# mean_absolute_error
print(mean_absolute_error(y_test,y_pred))

# mean_squared_log_error
print(mean_squared_log_error(y_test,y_pred))

# explained_variance_score
print(explained_variance_score(y_test,y_pred))






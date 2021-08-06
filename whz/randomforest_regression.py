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

regressor = RandomForestRegressor(n_estimators=150, random_state=1, max_depth=6, min_samples_split=3, min_impurity_decrease=0.1, warm_start=True)
# 交叉验证
cross_score = cross_val_score(regressor,X,y,cv=10,scoring="neg_mean_squared_error")

# 查看所有模型评估(打分)的列表
import sklearn
print("所有模型评估(打分)的列表:\n", sorted(sklearn.metrics.SCORERS.keys()))
scoring='neg_mean_squared_error'

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)


print("predict : \n", y_pred)
print("true : \n", y_test)
# 计算RMSE(均方根误差，标准误差)
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("R2_score", r2_score(y_test,y_pred))

# median_absolute_error
print("Median absolute error : ", median_absolute_error(y_test,y_pred))

# mean_absolute_error
print("mean_absolute_error : ", mean_absolute_error(y_test,y_pred))

# mean_squared_log_error
print("mean_squared_log_error : ", mean_squared_log_error(y_test,y_pred))

# explained_variance_score
print("explained_variance_score : ", explained_variance_score(y_test,y_pred))






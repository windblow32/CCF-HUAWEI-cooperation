import numpy as np
import btree as btree
import pandas as pd
import time

path = "E:\whz\LABDATA\\normal_s.csv"
data = pd.read_csv(path)
data = np.array(data).reshape(-1)
print("______________start BTree______________")
bt = btree.BTree(2)
start = time.time()
set_x = np.arange(len(data))
set_y = data.copy()
bt.build(set_x, set_y)
end = time.time()
print("B Tree build time: ", end - start)
print("=======================================")
err = 0
print("____________calculate error____________")
start = time.time()
for ind in range(len(set_x)):
    pre = bt.predict(set_x[ind])
    err += abs(pre - set_y[ind])
    if err != 0:
        flag = 1
        pos = pre
        off = 1
        while pos != set_y[ind]:
            pos += flag * off
            flag = -flag
            off += 1
end = time.time()
search = (end - start) / len(set_x)
print("B Tree search time: ", search)
mean_error = err * 1.0 / len(set_x)
print("mean error = ", mean_error)
print("=======================================")

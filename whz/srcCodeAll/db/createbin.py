import numpy as np
import pandas as pd
import csv
import random
a = np.arange(0, 100000000).reshape(-1)
# dataframe = pd.DataFrame({"num": a})
# start = time.time()
# dataframe.to_csv("normal.csv")
# end = time.time()
# print(end - start)
print("\n")
start = time.time()
a.tofile("bin.csv", sep="", )
end = time.time()
print(end - start)
b = np.fromfile("bin.csv", dtype=np.int32)
print(b)

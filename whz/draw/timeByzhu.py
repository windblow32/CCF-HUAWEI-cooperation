# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

name_list = ['85M', '140M', '280M', '359M']
# num_list = [970, 1100, 1060, 1021, 1008]
# num_list1 = [997.5, 997.5, 996, 997.4, 997.7]
# num_list2 = [1005.6, 1007, 1037, 1010.3, 997.4]
num_list = [310,260,221,208]
num_list1 = [197.5, 196, 197.4, 197.7]
num_list2 = [207, 237, 210.3, 197.4]
x = list(range(len(num_list)))
total_width, n = 0.8, 3
width = total_width / n

plt.bar(x, num_list,bottom=800, width=width, label='kraska')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, bottom=800, width=width, label='btree', tick_label=name_list,  fc='r')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, bottom=800, width=width, label='multi', fc='b')
plt.legend()
plt.savefig("E:/draw_picture/kraska,btree,multi/time_zhu.png")
plt.show()
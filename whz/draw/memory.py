from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

fig = plt.figure(figsize=(10, 7), dpi=100)
plt.rcParams['font.size'] = 13

kraska_x = np.array([
    # 18,
    34,
    75,
    85,
    140,
    200,
    280,
    350,
    762,
    1515,
    1945
])

kraska_y = np.array([
    # 382,
    382.6,
    383.4,
    385,
    384.3,
    384,
    384.6,
    241.6,
    341,
    384,
    384
])

# auto_x = np.array([
#      7,
#      700,
#      1515,
#      1945
# ])
# auto_y = np.array([
#     230,
#     273,
#     337,
#     300
#
# ])

btree_x = np.array([
    # 14,
    34,
    75,
    85,
    114,
    140,
    200,
    280
])

btree_y = np.array([
    # 369.7,
    805.2,
    1637.1,
    1898.8,
    2164.5,
    2657.9,
    4365.4,
    5723.7
])

multi_x = np.array([
    # 11.7,
    34,
    75,
    85,
    106,
    140,
    200,
    234,
    280

])

multi_y = np.array([
    # 90.3,
    100.7,
    141,
    137,
    132.6,
    65,
    30.6,
    129.7,
    213.6

])
plt.xlabel("size/Mb")
plt.ylabel("memory usage/Mb")
# plt.annotate('only use linear combination', xy=(350, 241.6), xytext=(500, 241.6), arrowprops=dict(arrowstyle='->'))

# kraska
# plt.annotate('(18, 382)', xy=(18, 382), xytext=(0, 1002),arrowprops=dict(arrowstyle='->'))
plt.annotate('(34, 382.6)', xy=(34, 382.6), xytext=(10, 412.6))
# plt.annotate('(75, 383.4)', xy=(75, 383.4), xytext=(55, 413.4))
plt.annotate('(85, 385)', xy=(85, 385), xytext=(65, 405))
plt.annotate('(140, 384.3)', xy=(140, 384.3), xytext=(140, 414.3))
plt.annotate('(200, 384)', xy=(200, 384), xytext=(200, 414))
plt.annotate('(280, 384.6)', xy=(280, 384.6), xytext=(260, 384.6))

# btree
plt.annotate('(34, 805.2)', xy=(34, 805.2), xytext=(10, 805.2))
# plt.annotate('(75, 1637.1)', xy=(75, 1637.1), xytext=(75, 1637.1))
plt.annotate('(85, 1898.8)', xy=(85, 1898.8), xytext=(95, 1888.8),arrowprops=dict(arrowstyle='->'))
plt.annotate('(140, 2657.9)', xy=(140, 2657.9), xytext=(140, 2657.9))
plt.annotate('(200, 4365.4)', xy=(200, 4365.4), xytext=(200, 4365.4))
# plt.annotate('(114, 2164.5)', xy=(114, 2164.5), xytext=(114, 2164.5))
plt.annotate('(280, 5723.1)', xy=(280, 5723.1), xytext=(260, 5723.1))

# multi
# plt.annotate('(11.7,90.3)', xy=(11.7, 90.3), xytext=(6.7, 120.3))
plt.annotate('(34, 100.7)', xy=(34, 100.7), xytext=(10.5,-100))
# plt.annotate('(75, 141)', xy=(75, 141), xytext=(60, -50))
plt.annotate('(85, 137)', xy=(85, 137), xytext=(65, -70))
# plt.annotate('(106, 132.6)', xy=(106, 132.6), xytext=(86, -100))
plt.annotate('(140, 65)', xy=(140, 65), xytext=(140, 65))
plt.annotate('(200, 30.6)', xy=(200, 30.6), xytext=(175, -100))
# plt.annotate('(234, 129.7)', xy=(234, 129.7), xytext=(234, -100))
plt.annotate('(280, 213.6)', xy=(280, 213.6), xytext=(265, 0))

# kraska
kraska_x_smooth = np.linspace(kraska_x.min(), kraska_x.max(), 300)
kraksa_y_smooth = make_interp_spline(kraska_x, kraska_y)(kraska_x_smooth)
plt.plot(kraska_x_smooth, kraksa_y_smooth, label='kraska')
plt.plot(kraska_x, kraska_y, 'om')

# btree
btree_x_smooth = np.linspace(btree_x.min(), btree_x.max(), 300)
btree_y_smooth = make_interp_spline(btree_x, btree_y)(btree_x_smooth)
plt.plot(btree_x_smooth, btree_y_smooth, label='btree', color='green')
plt.plot(btree_x, btree_y, 'og')

# multi
multi_x_smooth = np.linspace(multi_x.min(), multi_x.max(), 300)
multi_y_smooth = make_interp_spline(multi_x, multi_y)(multi_x_smooth)
plt.plot(multi_x_smooth, multi_y_smooth, label='multi-items', color='orange')
plt.plot(multi_x, multi_y, 'or')

plt.legend()

plt.xlim(xmin=0, xmax=300)
plt.savefig("E:/draw_picture/kraska,btree,multi/memory.png")
plt.show()

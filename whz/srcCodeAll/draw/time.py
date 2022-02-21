from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

fig = plt.figure(figsize=(10, 7), dpi=100)
plt.rcParams['font.size'] = 13
kraska_x = np.array([
    10,
    70,
    114,
    280,
    350
])

kraska_y = np.array([
    970,
    1007,
    1053,
    1121,
    1738
])
#
# auto_x = np.array([
#      7,
#      700,
#      1515,
#      1945
# ])
# auto_y = np.array([
#      912,
#      1000,
#      1200,
#      1300
# ])
btree_x = np.array([
    14,
    38,
    84,
    114,
    280,
    359
])
btree_y = np.array([
    1005.6,
    1002.3,
    997.5,
    994.5,
    997.4,
    1001
])
multi_x = np.array([
    11.7,
    33.5,
    143,
    195,
    280,
    359
])
multi_y = np.array([
    997.5,
    997.3,
    1037,
    996.3,
    1010.3,
    997.7
])

# kraska,multi
plt.annotate('(10, 970)', xy=(10, 970), xytext=(2, 930))
plt.annotate('(70, 1007)', xy=(70, 1007), xytext=(70, 1007))
plt.annotate('(114, 1100)', xy=(114, 1100), xytext=(114, 1100))
plt.annotate('(280, 1121)', xy=(280, 1121), xytext=(280, 1121))
plt.annotate('(350, 1738)', xy=(350, 1738), xytext=(350, 1738))

# auto
# plt.annotate('(7, 12)', xy=(7, 12), xytext=(7, 12))
# plt.annotate('(700, 1000)', xy=(700, 1000), xytext=(700, 1000))
# plt.annotate('(1515, 1200)', xy=(1515, 1300), xytext=(1515, 1200))
# plt.annotate('(1945, 1300)', xy=(1945, 1300), xytext=(1780, 1350))

# btree
plt.annotate('(14, 1005.6)', xy=(14, 1005.6), xytext=(5, 1100.6),arrowprops=dict(arrowstyle='->'))
plt.annotate('(38, 1002.3)', xy=(38, 1002.3), xytext=(38, 850),arrowprops=dict(arrowstyle='->'))
plt.annotate('(84, 997.5)', xy=(84, 997.5), xytext=(60, 1020.5))
plt.annotate('(114, 994.5)', xy=(114, 994.5), xytext=(90, 950.5))
plt.annotate('(280, 997.4)', xy=(280, 997.4), xytext=(280, 967.4))

# multi
plt.annotate('(11.7, 997.5)', xy=(11.7, 997.5), xytext=(2.7, 750.5),arrowprops=dict(arrowstyle='->'))
plt.annotate('(33.5, 997.3)', xy=(33.5, 997.3), xytext=(20, 1200),arrowprops=dict(arrowstyle='->'))
plt.annotate('(106, 1743.3)', xy=(106, 1743.3), xytext=(106, 1743.3))
plt.annotate('(143, 1037)', xy=(143, 1037), xytext=(143, 1037))
plt.annotate('(195, 996.3)', xy=(195, 996.3), xytext=(195, 996.3))
plt.annotate('(280, 1010.3)', xy=(280, 1010.3), xytext=(280, 1100.3))
plt.annotate('(359, 997.7)', xy=(359, 997.7), xytext=(359, 967.7))

plt.xlabel("size/Mb")
plt.ylabel("time/us")
# kraska,multi
kraska_x_smooth = np.linspace(kraska_x.min(), kraska_x.max(), 300)
kraska_y_smooth = make_interp_spline(kraska_x, kraska_y)(kraska_x_smooth)
plt.plot(kraska_x_smooth, kraska_y_smooth, label='kraska,multi')
plt.plot(kraska_x, kraska_y, 'om')

# plt.plot(kraska_x, kraska_y, label='kraska,multi')
# plt.plot(kraska_x, kraska_y, 'om')

# btree
btree_x_smooth = np.linspace(btree_x.min(), btree_x.max(), 300)
btree_y_smooth = make_interp_spline(btree_x, btree_y)(btree_x_smooth)
plt.plot(btree_x_smooth, btree_y_smooth, label='btree')
plt.plot(btree_x, btree_y, 'og')

# plt.plot(btree_x, btree_y, label='btree')
# plt.plot(btree_x, btree_y, 'og')

# multi todo
multi_x_smooth = np.linspace(multi_x.min(), multi_x.max(), 300)
multi_y_smooth = make_interp_spline(multi_x, multi_y)(multi_x_smooth)
plt.plot(multi_x_smooth, multi_y_smooth, label='multi')
plt.plot(multi_x, multi_y, 'or')

# plt.plot(multi_x, multi_y, label='multi')
# plt.plot(multi_x, multi_y, 'or')

# auto_x_smooth = np.linspace(auto_x.min(),auto_x.max(),300)
# auto_y_smooth = make_interp_spline(auto_x,auto_y)(auto_x_smooth)
# plt.plot(auto_x_smooth, auto_y_smooth, label='auto_model_selection')
plt.xlim(xmin=0,xmax=400)
plt.ylim(ymin=750,ymax=1300)

plt.legend()
plt.savefig("E:/draw_picture/kraska,multi,btree,multi/time.png")
plt.show()

from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

fig = plt.figure(figsize=(10, 8), dpi=100)
plt.rcParams['font.size'] = 13
# 由wiki 的 optimizer——out得来
kraska_x = np.array([
    6,
    12,
    24,
    96,
    192,
    384

])

kraska_y = np.array([
    0.99937,
    0.99942,
    0.99946,
    0.99952,
    0.99955,
    0.99957
])

multi_x = np.array([
    11.7,
    33.5,
    106,
    208
])
multi_y = np.array([
    0.9995584,
    0.9995583,
    0.9995583,
    0.9995893
])
#
#
# auto_x = np.array([
#     7,
#     700,
#     1515,
#     1945
# ])
# auto_y = np.array([
#     0.99634,
#     0.999564,
#     0.9996157,
#     0.9996325
#
# ])
# old kraska,multi:
# plt.annotate('(200, 0.99842)', xy=(200, 0.99842), xytext=(200, 0.99842))
# plt.annotate('(350, 0.99853)', xy=(350, 0.99853), xytext=(450, 0.99853))
# plt.annotate('(762, 0.999545)', xy=(762, 0.999545), xytext=(772, 0.999545))
# plt.annotate('(1515, 0.999604)', xy=(1515, 0.999604), xytext=(1315, 0.9995))
# plt.annotate('(1945, 0.999617)', xy=(1945, 0.999617), xytext=(1745, 0.999433))
# plt.annotate('(7, 12)', xy=(7, 12), xytext=(7, 12))

# new kraska,multi:
plt.annotate('(6,0.99937)', xy=(6,0.99937), xytext=(6,0.99937))
plt.annotate('(12, 0.99942)', xy=(12, 0.99942), xytext=(12, 0.99942))
plt.annotate('(24, 0.99946)', xy=(24, 0.99946), xytext=(24, 0.99946))
plt.annotate('(96, 0.99952)', xy=(96, 0.99952), xytext=(96, 0.99952))
plt.annotate('(192, 0.99955)', xy=(192, 0.99955), xytext=(192, 0.99955))
plt.annotate('(384, 0.99957)', xy=(384, 0.99957), xytext=(384, 0.99957))

# multi:
plt.annotate('(11.7, 0.9995584)', xy=(11.7, 0.999558435479), xytext=(11.7, 0.99957))
plt.annotate('(33.5, 0.9995583)', xy=(33.5, 0.9995583), xytext=(33.5, 0.99954))
plt.annotate('(106, 0.9995583)', xy=(106, 0.9995583), xytext=(106, 0.9995584))
plt.annotate('(208, 0.9995893)', xy=(208, 0.9995893), xytext=(208, 0.9995893))

# # auto nn
# plt.annotate('(700, 0.999564)', xy=(700, 0.999564), xytext=(400, 0.999564))
# plt.annotate('(1515, 0.999616)', xy=(1515, 0.999616), xytext=(1315, 0.99967))
# plt.annotate('(1945, 0.999633)', xy=(1945, 0.999633), xytext=(1745, 0.999717), arrowprops=dict(arrowstyle='->'))

plt.xlabel("size/Mb")
plt.ylabel("currency/us")
kraska_x_smooth = np.linspace(kraska_x.min(),kraska_x.max(),300)
kraska_y_smooth = make_interp_spline(kraska_x,kraska_y)(kraska_x_smooth)
plt.plot(kraska_x_smooth, kraska_y_smooth, label='kraska,multi')
plt.plot(kraska_x, kraska_y, 'om')

plt.xlim(xmin=0, xmax=400)

plt.ylim(ymin=0.99940, ymax=0.99970)

plt.plot(multi_x, multi_y, label='multi')
plt.plot(multi_x, multi_y, 'og')
plt.legend()
plt.savefig("E:/draw_picture/kraska,multi,btree,multi/currency.png")
plt.show()

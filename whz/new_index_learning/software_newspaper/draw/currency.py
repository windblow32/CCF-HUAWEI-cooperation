from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
fig = plt.figure(figsize=(10,8),dpi=100)
plt.rcParams['font.size'] = 13
kraska_x = np.array([
     200,
     350,
     762,
     1100,
     1515,
     1945

])

kraska_y = np.array([
    0.99842,
    0.99863,
    0.999545,
    0.999568,
    0.999604,
    0.999617

])

auto_x = np.array([
     10,
     700,
     1000,
     1515,
     1945
])
auto_y = np.array([
    0.99734,
    0.999654,
    0.999598,
    0.9996167,
    0.9996325

])

# plt.plot(200, 728, 'og')  # 绘制紫红色的圆形的点
# plt.plot(350, 1221, 'og')  # 绘制紫红色的圆形的点
# plt.plot(762, 1738, 'og')
# plt.plot(1515, 2158, 'og')
# plt.plot(1945, 3000, 'og')
#
#
# plt.plot(7, 12, 'om')
# plt.plot(700, 1000, 'om')
# plt.plot(1515, 1200, 'om')
# plt.plot(1945, 1300, 'om')

plt.annotate('(200, 0.99842)', xy=(200, 0.99842), xytext=(200, 0.99842))
plt.annotate('(350, 0.99853)', xy=(350, 0.99853), xytext=(450, 0.99853))
plt.annotate('(762, 0.999545)', xy=(762, 0.999545), xytext=(772, 0.999545))
plt.annotate('(1515, 0.999604)', xy=(1515, 0.999604), xytext=(1315, 0.9995))
plt.annotate('(1945, 0.999617)', xy=(1945, 0.999617), xytext=(1745, 0.999433))
# plt.annotate('(7, 12)', xy=(7, 12), xytext=(7, 12))
plt.annotate('(700, 0.999654)', xy=(700, 0.999654), xytext=(400, 0.999654))
plt.annotate('(1000, 0.999598)', xy=(1000, 0.999598), xytext=(1000, 0.999598))
plt.annotate('(1515, 0.999616)', xy=(1515, 0.999616), xytext=(1315, 0.99967))
plt.annotate('(1945, 0.999633)', xy=(1945, 0.999633), xytext=(1745, 0.999717),arrowprops=dict(arrowstyle='->'))

plt.xlabel("size/Mb")
plt.ylabel("currency/us")
kraska_x_smooth = np.linspace(kraska_x.min(),kraska_x.max(),300)


kraska_y_smooth = make_interp_spline(kraska_x,kraska_y)(kraska_x_smooth)
plt.plot(kraska_x_smooth, kraska_y_smooth, label='kraska,multi')
plt.plot(kraska_x, kraska_y, 'om')
auto_x_smooth = np.linspace(120,auto_x.max(),300)
plt.xlim(xmin=395, xmax=2000)

plt.ylim(ymin=0.99855, ymax=1)


auto_y_smooth = make_interp_spline(auto_x,auto_y)(auto_x_smooth)
plt.plot(auto_x_smooth, auto_y_smooth, label='auto_model_selection')
plt.plot(auto_x, auto_y, 'og')
plt.legend()
plt.savefig("E:/currency.png")
plt.show()
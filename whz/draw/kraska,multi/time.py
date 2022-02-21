from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
fig = plt.figure(figsize=(10,7),dpi=100)
plt.rcParams['font.size'] = 13
kraska_x = np.array([
     200,
     350,
     762,
     1515,
     1945

])

kraska_y = np.array([
     728,
     1221,
     1738,
     2158,
     3000

])

auto_x = np.array([
     7,
     700,
     1515,
     1945
])
auto_y = np.array([
     12,
     1000,
     1200,
     1300


])

plt.plot(200, 728, 'og')  # 绘制紫红色的圆形的点
plt.plot(350, 1221, 'og')  # 绘制紫红色的圆形的点
plt.plot(762, 1738, 'og')
plt.plot(1515, 2158, 'og')
plt.plot(1945, 3000, 'og')


plt.plot(7, 12, 'om')
plt.plot(700, 1000, 'om')
plt.plot(1515, 1200, 'om')
plt.plot(1945, 1300, 'om')

plt.annotate('(350, 241.6)', xy=(200, 728), xytext=(200, 728))
plt.annotate('(350, 1221)', xy=(350, 1221), xytext=(350, 1221))
plt.annotate('(762, 1738)', xy=(762, 1738), xytext=(762, 1738))
plt.annotate('(1515, 2158)', xy=(1515, 2000), xytext=(1515, 2158))
plt.annotate('(1945, 3000)', xy=(1945, 3000), xytext=(1780, 3000))
plt.annotate('(7, 12)', xy=(7, 12), xytext=(7, 12))
plt.annotate('(700, 1000)', xy=(700, 1000), xytext=(700, 1000))
plt.annotate('(1515, 1200)', xy=(1515, 1300), xytext=(1515, 1200))
plt.annotate('(1945, 1300)', xy=(1945, 1300), xytext=(1780, 1350))

plt.xlabel("size/Mb")
plt.ylabel("time/us")
kraska_x_smooth = np.linspace(kraska_x.min(),kraska_x.max(),300)


kraska_y_smooth = make_interp_spline(kraska_x,kraska_y)(kraska_x_smooth)
plt.plot(kraska_x_smooth, kraska_y_smooth, label='kraska,multi')

auto_x_smooth = np.linspace(auto_x.min(),auto_x.max(),300)


auto_y_smooth = make_interp_spline(auto_x,auto_y)(auto_x_smooth)
plt.plot(auto_x_smooth, auto_y_smooth, label='auto_model_selection')

plt.legend()
plt.savefig("E:/time.png")
plt.show()
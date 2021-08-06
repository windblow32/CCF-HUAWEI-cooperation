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
     12,
     250,
     350,
     500,
     700,
     1515,
     1945
])
auto_y = np.array([
     22,
     460,
     540,
     830,
     1000,
     1757,
     2300


])

plt.plot(200, 728, 'og')  # 绘制紫红色的圆形的点
plt.plot(350, 1221, 'og')  # 绘制紫红色的圆形的点
plt.plot(762, 1738, 'og')
plt.plot(1515, 2158, 'og')
plt.plot(1945, 3000, 'og')


plt.plot(12, 22, 'om')
plt.plot(250, 460, 'om')
plt.plot(350, 540, 'om')
plt.plot(500, 830, 'om')
plt.plot(700, 1000, 'om')
plt.plot(1515, 1757, 'om')
plt.plot(1945, 2300, 'om')

plt.annotate('(200, 241.6)', xy=(200, 728), xytext=(150, 728))
plt.annotate('(350, 1221)', xy=(350, 1221), xytext=(350, 1221))
plt.annotate('(762, 1738)', xy=(762, 1738), xytext=(762, 1738))
plt.annotate('(1515, 2158)', xy=(1515, 2000), xytext=(1515, 2158))
plt.annotate('(1945, 3000)', xy=(1945, 3000), xytext=(1780, 3000))
plt.annotate('(12, 22)', xy=(12, 22), xytext=(12, 22))
plt.annotate('(250, 460)', xy=(250, 460), xytext=(200, 360))
plt.annotate('(500, 830)', xy=(500, 830), xytext=(500, 830))
plt.annotate('(350, 540)', xy=(350, 540), xytext=(350, 540))
plt.annotate('(700, 1000)', xy=(700, 1000), xytext=(700, 1000))
plt.annotate('(1515, 1757)', xy=(1515, 1757), xytext=(1515, 1757))
plt.annotate('(1945, 2300)', xy=(1945, 2300), xytext=(1780, 2350))

plt.xlabel("size/Mb")
plt.ylabel("time/us")
kraska_x_smooth = np.linspace(kraska_x.min(),kraska_x.max(),300)


kraska_y_smooth = make_interp_spline(kraska_x,kraska_y)(kraska_x_smooth)
plt.plot(kraska_x_smooth, kraska_y_smooth, label='kraska')

auto_x_smooth = np.linspace(auto_x.min(),auto_x.max(),300)


auto_y_smooth = make_interp_spline(auto_x,auto_y)(auto_x_smooth)
plt.plot(auto_x_smooth, auto_y_smooth, label='auto_model_selection')

plt.legend()
plt.savefig("E:/time.png")
plt.show()
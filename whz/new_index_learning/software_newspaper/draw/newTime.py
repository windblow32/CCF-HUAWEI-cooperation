from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

fig = plt.figure(figsize=(10, 7), dpi=100)
plt.rcParams['font.size'] = 13
kraska_x = np.array([
    350,
    700,
    750,
    800,
    1000,
    1515,
    1945

])

kraska_y = np.array([
    1221,
    1515,
    1738,
    1755,
    1919,
    2158,
    3000

])

auto_x = np.array([
    350,
    700,
    750,
    800,
    1000,
    1515,
    1945
])
auto_y = np.array([
    1210,
    882,
    785,
    656,
    1413,
    2120,
    2427

])

plt.annotate('(350, 1210)', xy=(350, 1210), xytext=(350, 1410))
plt.annotate('(350, 1221)', xy=(350, 1221), xytext=(350, 1021))
plt.annotate('(700, 882)', xy=(700, 882), xytext=(600, 1082))
plt.annotate('(700, 1515)', xy=(700, 1515), xytext=(700, 1515))
plt.annotate('(750, 785)', xy=(750, 785), xytext=(700, 785))
plt.annotate('(750, 1738)', xy=(750, 1738), xytext=(700, 1638))
plt.annotate('(800, 656)', xy=(800, 656), xytext=(800, 656))
plt.annotate('(800, 1755)', xy=(800, 1755), xytext=(800, 1855))
plt.annotate('(1000,1413)', xy=(1000, 1413), xytext=(1000, 1413))
plt.annotate('(1000, 1919)', xy=(1000, 1919), xytext=(1000, 1919))
plt.annotate('(1515,2120)', xy=(1515, 2120), xytext=(1515, 2020))
plt.annotate('(1515,2158)', xy=(1515, 2120), xytext=(1515, 2258))
plt.annotate('(1945,3000)', xy=(1945, 3000), xytext=(1945, 3000))
plt.annotate('(1945,2427)', xy=(1945, 2427), xytext=(1945, 2427))


plt.xlabel("size/MB")
plt.ylabel("time/us")
# kraska_x_smooth = np.linspace(kraska_x.min(), kraska_x.max(), 2000)
#
# kraska_y_smooth = make_interp_spline(kraska_x, kraska_y)(kraska_x_smooth)
plt.plot(kraska_x, kraska_y, label='kraska,multi', linewidth=2.0, linestyle='--')
plt.plot(kraska_x, kraska_y, 'om')

# auto_x_smooth = np.linspace(auto_x.min(), auto_x.max(), 300)

# auto_y_smooth = make_interp_spline(auto_x, auto_y)(auto_x_smooth)
plt.plot(auto_x, auto_y, label='auto_model_selection', linewidth=2.0, linestyle='--')
plt.plot(auto_x, auto_y, 'og')

plt.legend()
plt.savefig(r"E:/draw_picture/auto,kraska,multi/time.png")
plt.show()

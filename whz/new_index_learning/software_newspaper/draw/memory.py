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
     1900

])

kraska_y = np.array([
    284,
    241.6,
    341,
    584,
    740

])

auto_x = np.array([
     7,
     700,
     1515,
     1900
])
auto_y = np.array([
    230,
    273,
    537,
    600

])
plt.xlabel("size/Mb")
plt.ylabel("memory usage/Mb")
plt.annotate('only use linear combination', xy=(350, 241.6), xytext=(350, 441.6), arrowprops=dict(arrowstyle='->'))

plt.annotate('(200, 284)', xy=(200, 284), xytext=(200, 284))
plt.annotate('(350, 241.6)', xy=(350, 241.6), xytext=(350, 231.6))
plt.annotate('(762, 341)', xy=(762, 341), xytext=(762, 341))
plt.annotate('(1515, 584)', xy=(1515, 584), xytext=(1515, 584))
plt.annotate('(1900, 740)', xy=(1900, 740), xytext=(1900, 740))
plt.annotate('(7, 230)', xy=(7, 230), xytext=(7, 230))
plt.annotate('(700, 273)', xy=(700, 273), xytext=(700, 273))
plt.annotate('(1515, 537)', xy=(1515, 537), xytext=(1515, 537))
plt.annotate('(1900, 600)', xy=(1900, 600), xytext=(1900, 600))
# kraska_x_smooth = np.linspace(kraska_x.min(),kraska_x.max(),300)
# auto_x_smooth = np.linspace(auto_x.min(),auto_x.max(),300)
plt.plot(kraska_x, kraska_y, label='kraska')
plt.plot(kraska_x, kraska_y, 'om')
plt.plot(auto_x, auto_y, label='auto_model_selection')
plt.plot(auto_x, auto_y, 'og')
plt.legend()

# kraska_x_smooth = np.linspace(kraska_x.min(),kraska_x.max(),300)
#
#
# kraska_y_smooth = make_interp_spline(kraska_x,kraska_y)(kraska_x_smooth)
# plt.plot(kraska_x_smooth, kraska_y_smooth)
#
# auto_x_smooth = np.linspace(auto_x.min(),auto_x.max(),300)
#
#
# auto_y_smooth = make_interp_spline(auto_x,auto_y)(auto_x_smooth)
# plt.plot(auto_x_smooth, auto_y_smooth)
plt.savefig("E:/memory.png")
plt.show()
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
fig = plt.figure(figsize=(10,7),dpi=100)
plt.rcParams['font.size'] = 13
kraska_x = np.array([
    700,
    750,
    1000,
    1515,
    1945
])

kraska_y = np.array([
    384,
    384,
    384,
    384,
    384
])

auto_x = np.array([
    700,
    750,
    1000,
    1515,
    1945
])
auto_y = np.array([
    339,
    354,
    259,
    338,
    320
])
plt.xlabel("size/MB")
plt.ylabel("memory usage/MB")

# plt.annotate('(350, 241)', xy=(350, 241), xytext=(350, 241))
# plt.annotate('(350, 411)', xy=(350, 411), xytext=(350, 411))
plt.annotate('(750, 384)', xy=(750, 384), xytext=(800, 374))
plt.annotate('(750, 354)', xy=(750, 354), xytext=(750, 354))
plt.annotate('(1000, 384)', xy=(1000, 384), xytext=(1000, 384))
plt.annotate('(1000, 259)', xy=(1000, 259), xytext=(1000, 259))
plt.annotate('(1515, 384)', xy=(1515, 384), xytext=(1515, 384))
plt.annotate('(1515, 338)', xy=(1515, 338), xytext=(1515, 338))
plt.annotate('(1945, 384)', xy=(1945, 384), xytext=(1945, 384))
plt.annotate('(1945, 320)', xy=(1945, 320), xytext=(1945, 320))
plt.annotate('(700, 384)', xy=(700, 384), xytext=(650, 384))
plt.annotate('(700, 339)', xy=(700, 339), xytext=(700, 339))
# plt.annotate('(800, 384)', xy=(800, 384), xytext=(800, 384))
# plt.annotate('(800, 142)', xy=(800, 142), xytext=(800, 142))


plt.plot(kraska_x, kraska_y, label='kraska,multi', linewidth=2.0, linestyle='--')
plt.plot(kraska_x, kraska_y, 'om')
plt.plot(auto_x, auto_y, label='auto_model_selection', linewidth=2.0, linestyle='--')
plt.plot(auto_x, auto_y, 'og')
plt.legend()

plt.savefig(r"E:/draw_picture/auto,kraska,multi/memory.png")
plt.show()
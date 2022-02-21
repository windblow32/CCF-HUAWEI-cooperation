from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

fig = plt.figure(figsize=(10, 8), dpi=100)
plt.rcParams['font.size'] = 13
# 由wiki 的 optimizer——out得来
kraska_x = np.array([
    700,
    800,
    1000,
    1515,
    1945
])

kraska_y = np.array([
    0.999561,
    0.999564,
    0.999596,
    0.999604,
    0.999617
])

multi_x = np.array([
    700,
    800,
    1000,
    1515,
    1945
])
multi_y = np.array([
    0.9995583,
    0.9995893,
    0.999785,
    0.9997840,
    0.9997844
])

# new kraska,multi:
# plt.annotate('(350,0.99853)', xy=(350, 0.99853), xytext=(350, 0.99853))
plt.annotate('(700,0.999561)', xy=(700, 0.999561), xytext=(700, 0.999561))
# plt.annotate('(750, 0.999564)', xy=(750, 0.999564), xytext=(750, 0.999564))
plt.annotate('(800,0.999564)', xy=(800, 0.999564), xytext=(800, 0.99957))
plt.annotate('(1000,0.999596)', xy=(1000, 0.999596), xytext=(1000, 0.99960))
plt.annotate('(1515,0.999604)', xy=(1515, 0.999604), xytext=(1515, 0.99961))
plt.annotate('(1945,0.999617)', xy=(1945, 0.999617), xytext=(1945, 0.999617))

# multi:
# plt.annotate('(350, 0.9995584)', xy=(350, 0.9995584), xytext=(350, 0.9995584))
plt.annotate('(700,0.9995583)', xy=(700, 0.9995583), xytext=(700, 0.999553))
# plt.annotate('(750,0.9995583)', xy=(750, 0.9995583), xytext=(750, 0.9995583))
plt.annotate('(800,0.9995893)', xy=(800, 0.9995893), xytext=(800, 0.9995893))
plt.annotate('(1000,0.999785)', xy=(1000, 0.999785), xytext=(1000, 0.999785))
plt.annotate('(1515,0.9997840)', xy=(1515, 0.9997840), xytext=(1515, 0.9997840))
plt.annotate('(1945,0.9997844)', xy=(1945, 0.9997844), xytext=(1945, 0.9997844))

plt.xlabel("size/MB")
plt.ylabel("currency/us")
plt.plot(kraska_x, kraska_y, label='kraska,multi', linewidth=2.0, linestyle='--')
plt.plot(kraska_x, kraska_y, 'om')

plt.plot(multi_x, multi_y, label='auto_model_selection', linewidth=2.0, linestyle='--')
plt.plot(multi_x, multi_y, 'og')
plt.legend()
plt.savefig(r"E:/draw_picture/auto,kraska,multi/currency.png")
plt.show()

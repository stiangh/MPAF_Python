import matplotlib.pyplot as plt
import numpy as np
from hsi_io import ABU_FILES, import_hsi, pixel_plot
from stats import normalize, entropy
from mpaf import band_selection, MPAF, band_to_anomaly_score
from evaluation import ROC

# filename = ABU_FILES[12]

# _, hsi, _ = import_hsi(filename)

# ys = []
# xs = []
# for i, band in enumerate(hsi):
#     ys.append(entropy(band))
#     xs.append(i)

# plt.figure()
# plt.plot(xs, ys)
# plt.show()

# fig_nr = 1
# for filename in ABU_FILES:
#     name = filename[9:-4]
#     if name == 'abu-urban-2':
#         [x_dim, y_dim, s_dim], hsi, truth = import_hsi(filename)

#         band, _ = band_selection(hsi, s_dim, 10, 4, 0.15, 0.04)

#         plt.subplot(2, 3, fig_nr)
#         plt.imshow(band)
#         plt.title(name)

#         plt.subplot(2, 3, 3+fig_nr)
#         plt.imshow(truth, cmap='Greys_r')
#         plt.title(name)

#         fig_nr += 1

# plt.show()

# sel_band, BA = band_selection(hsi, s_dim, 10, 4, 0.15, 0.04)

# for i, band in enumerate(hsi):
#     band_norm = normalize(band)
#     if (band_norm == sel_band).all():
#         print(f'Selected band #: {i}')

t, u, alpha, beta = 10, 4, 0.15, 0.04
se2, se3 = 3, 3

for i, filename in enumerate(ABU_FILES[:4]):
    name = filename[9:-4]

    print(f'name: {name}')

    [x_dim, y_dim, s_dim], hsi, truth = import_hsi(filename)

    O = MPAF(hsi, s_dim, t, u, alpha, beta, se2, se3)

    # band, BA = band_selection(hsi, s_dim, t, u, alpha, beta)

    # O, _, _ = band_to_anomaly_score(band, BA, 3, se2, se3, 25)

    # AUC, TPRS, FPRS = ROC(O, truth)

    plt.subplot(2, 4, i+1)
    plt.imshow(O, cmap='Greys_r')
    plt.title(f'{name}: Output')

    plt.subplot(2, 4, i+5)
    plt.imshow(truth, cmap='Greys_r')
    plt.title(f'{name}: Reference')

    # plt.subplot(1, 5, i+1)
    # plt.plot(FPRS, TPRS)
    # plt.title(f'{name}: ROC')

    # print(f'AUC:  {AUC}')

plt.show()

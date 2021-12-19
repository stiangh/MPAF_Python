import numpy as np
import matplotlib.pyplot as plt

from hsi_io import import_hsi, ABU_FILES
from mpaf import area_selection, band_to_anomaly_score
from evaluation import ROC
from stats import normalize

# filename = ABU_FILES[9]
# name = filename[9:-4]

# [x_dim, y_dim, s_dim], hsi, truth = import_hsi(filename)

out_filename = 'res/best_bands_3_3.csv'

t, u, alpha, beta = 10, 4, 0.15, 0.04
se2, se3 = 3, 3

data = [['name', 'best_band', 'AUC']]

# with open(out_filename, 'w') as file:
#     file.write(','.join([str(x) for x in data[0]])+'\n')

for filename in ABU_FILES[4:]:
    name = filename[9:-4]

    [x_dim, y_dim, s_dim], hsi, truth = import_hsi(filename)

    best_AUC = 0
    best_band = None
    # AUC_by_band = np.zeros(s_dim)
    for i in range(s_dim):
        band = hsi[i]

        print(f'{name}: Working on band {i+1} of {s_dim}...')

        band_norm = normalize(band)

        n_extreme_brights = np.sum( band_norm >= (1 - alpha) )
        n_extreme_darks = np.sum( band_norm <= alpha )

        if n_extreme_brights > n_extreme_darks:
            BA = True
        else:
            BA = False

        k, se1 = area_selection(band_norm, BA)

        O, _, _ = band_to_anomaly_score(band, BA, se1, se2, se3, k)

        AUC, _, _ = ROC(O, truth)

        # AUC_by_band[i] = AUC

        if AUC > best_AUC:
            best_AUC = AUC
            best_band = i

    data.append([name, best_band, best_AUC])

    with open(out_filename, 'a') as file:
        file.write(','.join([str(x) for x in [name, best_band, best_AUC]])+'\n')

print(data)


# print()
# print(f'Best band: {best_band}')
# print(f'Best AUC:  {best_AUC}')

# plt.figure()
# plt.plot([i for i in range(s_dim)], AUC_by_band)
# plt.show()

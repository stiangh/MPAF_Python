from typing import Iterator
import numpy as np
from mpaf import band_selection, band_to_anomaly_score
from evaluation import ROC
from hsi_io import import_hsi, ABU_FILES

filename = ABU_FILES[11]
name = filename[9:-4]

[x_dim, y_dim, s_dim], hsi, truth = import_hsi(filename)

t, u, alpha, beta = 10, 4, 0.15, 0.04
band, BA = band_selection(hsi, s_dim, t, u, alpha, beta)

se1_start = 3
se2_start = 1
se3_start = 3
k_start = 25

se1_possibles = [3, 5, 7, 9]
se2_possibles = [1, 3, 5, 7]
se3_possibles = [1, 3, 5, 7]
k_possibles = [15, 20, 25, 30]

AUC_best = 0
se1_best = None
se2_best = None
se3_best = None
k_best = None

iterations = 0
for se1 in se1_possibles:
    for se2 in se2_possibles:
        for se3 in se3_possibles:
            for k in k_possibles:
                O, _, _ = band_to_anomaly_score(band, BA, se1, se2, se3, k)
                AUC, _, _ = ROC(O, truth)

                if AUC > AUC_best:
                    AUC_best = AUC
                    se1_best, se2_best, se3_best, k_best = se1, se2, se3, k

                iterations += 1
                print(f'Iteration {iterations}: se1={se1},se2={se2},se3={se3},k={k},AUC={np.round(AUC*10000)/10000}')

print()
print(f'name:\t\t{name}')
print(f'best se1:\t{se1_best}')
print(f'best se2:\t{se2_best}')
print(f'best se3:\t{se3_best}')
print(f'best k:\t\t{k_best}')
print(f'best AUC:\t{AUC_best}')

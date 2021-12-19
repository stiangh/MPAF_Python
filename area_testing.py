import numpy as np
from hsi_io import import_hsi, ABU_FILES
from mpaf import band_selection
from stats import normalize, otsu

otsus = []
anomaly_levels_by_file = []
for i, filename in enumerate(ABU_FILES):
    [x_dim, y_dim, s_dim], hsi, truth = import_hsi(filename)

    band, BA = band_selection(hsi, s_dim, 100, 4, 0.15, 0.04)

    band_norm = normalize(band)

    if not BA:
        band_norm = 1 - band_norm
    
    anomaly_levels = []
    for y in range(y_dim):
        for x in range(x_dim):
            if truth[y][x] == True:
                anomaly_levels.append(band_norm[y][x])
    
    anomaly_levels_by_file.append(anomaly_levels)

    o = otsu(band_norm)
    otsus.append(o)

averages = []

for i, anomaly_levels in enumerate(anomaly_levels_by_file):
    name = ABU_FILES[i][9:-4]
    m = np.mean(anomaly_levels)
    s = np.std(anomaly_levels)
    min_val = np.min(anomaly_levels)
    max_val = np.max(anomaly_levels)

    print(f'name:   {name}')
    print(f'  mean: {m}')
    print(f'  std:  {s}')
    print(f'  min:  {min_val}')
    print(f'  max:  {max_val}')
    print(f'  otsu: {otsus[i]}')
    print()

import numpy as np

from hsi_io import import_hsi, ABU_FILES
from mpaf import MPAF
from evaluation import ROC

for se2 in [1, 3, 5, 7]:
    for se3 in [1, 3, 5, 7]:
        print(f'se2={se2},se3={se3}')
        out_filename = f'res/AUC_w_area_sel_{se2}_{se3}.csv'

        data = [['name', 'AUC']]
        for filename in ABU_FILES:
            name = filename[9:-4]
            [x_dim, y_dim, s_dim], hsi, truth = import_hsi(filename)

            print(f'Working on {name}...')

            O = MPAF(hsi, s_dim, 10, 4, 0.15, 0.04, se2, se3)

            AUC, _, _ = ROC(O, truth)

            data.append([name, np.round(AUC*10000)/10000])

        airport_data = []
        beach_data = []
        urban_data = []
        for line in data:
            name, auc = line
            if 'airport' in name:
                airport_data.append(auc)
            elif 'beach' in name:
                beach_data.append(auc)
            elif 'urban' in name:
                urban_data.append(auc)

        airport_avg = np.mean(airport_data)
        beach_avg = np.mean(beach_data)
        urban_avg = np.mean(urban_data)
        scene_avg = np.mean([airport_avg, beach_avg, urban_avg])

        data.append(['airport-avg', airport_avg])
        data.append(['beach-avg', beach_avg])
        data.append(['urban-avg', urban_avg])
        data.append(['scene-avg', scene_avg])

        lines = [','.join([str(x) for x in arr])+'\n' for arr in data]
        print(lines)

        with open(out_filename, 'w') as file:
            file.writelines(lines)

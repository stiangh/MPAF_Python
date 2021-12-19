import numpy as np

def ROC(O, truth, granularity=100):
    y_dim, x_dim = np.shape(O)

    TPRS = []
    FPRS = []
    tresholds = [x/granularity for x in range(granularity+1)]
    for t in tresholds:
        TP, TN, FP, FN = 0, 0, 0, 0
        anomalies = O >= t
        
        for y in range(y_dim):
            for x in range(x_dim):
                if anomalies[y][x] == True:
                    if truth[y][x] == True:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if truth[y][x] == True:
                        FN += 1
                    else:
                        TN += 1
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TPRS.append(TPR)
        FPRS.append(FPR)

    AUC = - np.trapz(TPRS, FPRS)

    return AUC, TPRS, FPRS

if __name__ == "__main__":
    from hsi_io import import_hsi, ABU_FILES
    from mpaf import band_selection, band_to_anomaly_score

    data = [['name', 'AUC']]
    for filename in ABU_FILES:
        name = filename[9:-4]
        print(f'Working on {name}...')

        [x_dim, y_dim, s_dim], hsi, truth = import_hsi(filename)

        t, u, alpha, beta = 10, 4, 0.15, 0.04
        band, BA = band_selection(hsi, s_dim, t, u, alpha, beta)

        se1, se2, se3, k = 3, 7, 1, 20
        O, _, _ = band_to_anomaly_score(band, BA, se1, se2, se3, k)

        AUC, _, _ = ROC(O, truth)

        data.append([filename[9:-4], np.round(AUC*10000)/10000])
    
    lines = [','.join([str(x) for x in arr])+'\n' for arr in data]
    print(lines)

    with open('res/AUC_3_7_1_20.csv', 'w') as file:
        file.writelines(lines)

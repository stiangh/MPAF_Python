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

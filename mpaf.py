import numpy as np

from hsi_io import import_hsi, ABU_FILES
from stats import normalize, entropy, otsu
from attribute import tresh, connected_components
from morph import erosion, dilation, opening, closing

import matplotlib.pyplot as plt

def diff_map(norm_band, BA, k):
    x_dim, y_dim = np.shape(norm_band)

    if BA:
        thinned_image = np.zeros((x_dim, y_dim))
        grayscale_levels = [x/50 for x in range(1, 51)]
        for gray_level in grayscale_levels:
            bin_image = tresh(norm_band, gray_level, geq=True)
            _, area_image = connected_components(bin_image)
            comp_removed_image = bin_image & (area_image > k)

            for y in range(y_dim):
                for x in range(x_dim):
                    if comp_removed_image[y][x] == True:
                        thinned_image[y][x] = gray_level

        D = norm_band - thinned_image
    else:
        thickened_image = np.ones((x_dim, y_dim))
        grayscale_levels = [x/50 for x in range(1, 51)]
        for gray_level in reversed(grayscale_levels):
            bin_image = tresh(norm_band, gray_level, geq=False)
            _, area_image = connected_components(bin_image)
            comp_removed_image = bin_image & (area_image > k)

            for y in range(y_dim):
                for x in range(x_dim):
                    if comp_removed_image[y][x] == True:
                        thickened_image[y][x] = gray_level
        
        D = thickened_image - norm_band

    return D

def band_to_anomaly_score(norm_band, BA, se1, se2, se3, k):

    if BA:
        R = dilation((norm_band - opening(norm_band, se1)), se2)
    else:
        R = dilation((closing(norm_band, se1) - norm_band), se2)

    D = diff_map(norm_band, BA, k)

    O = R * dilation(D, se3)

    # Normalize the anomaly scores 
    min_val = np.min(O)
    max_val = np.max(O)
    O = (O - min_val) / (max_val - min_val)

    return O, R, D

def band_selection(hsi, s_dim, t, u, alpha, beta):
    sampled_indexes = []
    sampled_bands = []
    k = 0
    while (k*t+u) < s_dim:
        sampled_indexes.append(k*t+u)
        sampled_bands.append(hsi[k*t+u])
        k += 1
    
    BAs = []
    DAs = []
    for band in sampled_bands:
        band_norm = normalize(band)

        n_extreme_brights = np.sum( band_norm >= (1 - alpha) )
        n_extreme_darks = np.sum( band_norm <= alpha )

        if n_extreme_brights > n_extreme_darks:
            BAs.append(band_norm)
        else:
            DAs.append(band_norm)

    if len(BAs) > len(DAs):
        BA = True
        XC = BAs
    else:
        BA = False
        XC = DAs

    entropies = np.zeros(s_dim)
    for i, band in enumerate(hsi):
        band_norm = normalize(band)
        H = entropy(band_norm)
        entropies[i] = H
    H_m = np.mean(entropies)
    H_s = np.std(entropies)

    XH = []
    for band_norm in XC:
        # band_norm = normalize(band)
        H = entropy(band_norm)
        if H > (H_m - 2*H_s):
            XH.append(band_norm)

    if BA:
        min_sum = float('inf')
        min_sum_band = None
        for band_norm in XH:
            # band_norm = normalize(band)
            curr_sum = np.sum( band_norm >= (0.5 - beta) )
            if curr_sum < min_sum:
                min_sum = curr_sum
                min_sum_band = band_norm
        return min_sum_band, BA
    else:
        min_sum = float('inf')
        min_sum_band = None
        for band_norm in XH:
            # band_norm = normalize(band)
            curr_sum = np.sum( band_norm <= (0.5 + beta) )
            if curr_sum < min_sum:
                min_sum = curr_sum
                min_sum_band = band_norm
        return min_sum_band, BA

def area_selection(norm_band, BA):
    x_dim, y_dim = np.shape(norm_band)

    N_pixels = x_dim * y_dim
    k = N_pixels / 100
    
    D = diff_map(norm_band, BA, k)

    otsu_level = otsu(D)

    BW = D >= otsu_level

    id_image, area_image = connected_components(BW)

    edges_table = {}
    area_table = {}

    # Find bounding boxes for each component
    for y in range(y_dim):
        for x in range(x_dim):
            if id_image[y][x] > 0:
                #pixel is part of connected component
                pixel_id = id_image[y][x]
                if not pixel_id in edges_table:
                    edges_table[pixel_id] = {'top': y, 'bottom': y, 'left': x, 'right': x}
                    area_table[pixel_id] = area_image[y][x]
                else:
                    edges_table[pixel_id]['left'] = min(x, edges_table[pixel_id]['left'])
                    edges_table[pixel_id]['right'] = max(x, edges_table[pixel_id]['right'])
                    edges_table[pixel_id]['bottom'] = max(y, edges_table[pixel_id]['bottom'])

    areas = [x for x in area_table.values()]
    area_mean = np.mean(areas)
    area_std = np.std(areas)
    area_max = np.max(areas)

    if area_max <= area_mean + 2 * area_std:
        A_k = area_max
    else:
        min_big_area = N_pixels
        for area in areas:
            if area >= area_mean + 2 * area_std:
                if area < min_big_area:
                    min_big_area = area
        A_k = min_big_area
    
    new_k = np.power(np.sqrt(A_k), 2)

    # Find the largest edge from bounding boxes
    largest_edge = 0
    for id, area in area_table.items():
        if area == A_k:
            width = np.abs(edges_table[id]['right'] - edges_table[id]['left'])
            height = np.abs(edges_table[id]['top'] - edges_table[id]['bottom'])
            large_edge = max(width, height)
            if large_edge > largest_edge:
                largest_edge = large_edge
    
    new_se1 = min(largest_edge, np.sqrt(N_pixels/100))

    return int(new_k), int(new_se1)

def MPAF(hsi, s_dim, t, u, alpha, beta, se2, se3):
    band_norm, BA = band_selection(hsi, s_dim, t, u, alpha, beta)

    k, se1 = area_selection(band_norm, BA)

    O, _, _ = band_to_anomaly_score(band_norm, BA, se1, se2, se3, k)

    return O

if __name__ == "__main__":
    filename = ABU_FILES[0]
    [x_dim, y_dim, s_dim], hsi, truth = import_hsi(filename)
    
    print(f'name:\t{filename[9:-4]}')

    band, BA = band_selection(hsi, s_dim, 10, 5, 0.15, 0.04)

    se1, se2, se3, k = 3, 3, 5, 15

    O, R, D = band_to_anomaly_score(band, BA, se1, se2, se3, k)

    plt.figure(1)

    plt.subplot(2, 3, 1)
    plt.imshow(band)
    plt.title('Original')

    plt.subplot(2, 3, 2)
    plt.imshow(R)
    plt.title('MP')

    plt.subplot(2, 3, 3)
    plt.imshow(D)
    plt.title('Diff Map')

    plt.subplot(2, 3, 4)
    plt.imshow(O, cmap='Greys_r')
    plt.title('Output')

    plt.subplot(2, 3, 6)
    plt.imshow(truth, cmap='Greys_r')
    plt.title('Reference')

    TPRS = []
    FPRS = []
    best_tresh = None
    best_perf = 0
    best_perf_tpr = None
    best_perf_fpr = None
    for t in [x/100 for x in range(0, 101)]:
        anomalies = O >= t
        TP = 0
        TN = 0
        FP = 0
        FN = 0
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
        perf = TP+TN
        if perf > best_perf:
            best_perf = perf
            best_tresh = t
            best_tresh_tpr = TPR
            best_tresh_fpr = FPR

    plt.subplot(2, 3, 5)
    plt.imshow(O > best_tresh, cmap='Greys_r')
    plt.title('Best treshold')
    
    plt.figure(2)
    plt.plot(FPRS, TPRS)

    plt.show()

    AUC = - np.trapz(TPRS, FPRS)
    print(f'AUC:\t\t{AUC}')
    print(f'best_tresh:\t\t{best_tresh}')
    print(f'best_tresh_fpr:\t{best_tresh_fpr}')
    print(f'best_tresh_tpr:\t{best_tresh_tpr}')

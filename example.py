import numpy as np
import matplotlib.pyplot as plt

from hsi_io import import_hsi, ABU_FILES
from mpaf import MPAF
from evaluation import ROC

filename = ABU_FILES[0] # airport-1
name = filename[9:-4]

# read the HSI CSV file
[x_dim, y_dim, s_dim], hsi, truth = import_hsi(filename)

# choose parameters to MPAF
t, u, alpha, beta, se2, se3 = 10, 4, 0.15, 0.04, 3, 3

# Perform the algorithm to get anomaly scores
O = MPAF(hsi, s_dim, t, u, alpha, beta, se2, se3)

# Calculate ROC
AUC, TPRS, FPRS = ROC(O, truth)

# Display HSI, anomaly score map and truth map side by side
plt.figure()

band_nr = 100 # Showing band nr 100
plt.subplot(1, 3, 1)
plt.title(f"HSI, band_nr:{band_nr}")
plt.imshow(hsi[band_nr])

plt.subplot(1, 3, 2)
plt.title("Anomaly Scores")
plt.imshow(O, cmap='Greys_r')

plt.subplot(1, 3, 3)
plt.title("Reference")
plt.imshow(truth, cmap='Greys_r')

# Display ROC and print AUC
a = plt.figure()
a.add_axes()
plt.title("ROC")
plt.plot(FPRS, TPRS)

print(f"name:\t{name}")
print(f"AUC:\t{AUC}")

plt.show()

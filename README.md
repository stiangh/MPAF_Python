# README

This repository contains the python implementation of the MPAF HAD algorithm, developed as part of my specialization project (TFE4580) at NTNU. The file `mpaf.py` contains the function `MPAF`, which performs the algorithm. The file `evaluation.py` contains the function `ROC`, used for calculating ROC, including AUC. The file `hsi_io.py` contains the function `import_hsi`, which reads an HSI from a CSV file, and the list `ABU_FILES`, which contain the filepaths of the dataset. Example usage is shown in the file `example.py`, and this file also shows how to display the results using matplotlib.

The ABU dataset was downloaded from http://xudongkang.weebly.com/data-sets.html, and is stored in the folder `data/mat`. The matlab script `mat_to_csv.m` was used to convert the dataset into CSV files, and these CSV files are stored in the folder `data/csv`.

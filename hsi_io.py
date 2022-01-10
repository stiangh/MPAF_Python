import numpy as np
import matplotlib.pyplot as plt

ABU_FILES = [
    'data/csv/abu-airport-1.csv',
    'data/csv/abu-airport-2.csv',
    'data/csv/abu-airport-3.csv',
    'data/csv/abu-airport-4.csv',
    'data/csv/abu-beach-1.csv',
    'data/csv/abu-beach-2.csv',
    'data/csv/abu-beach-3.csv',
    'data/csv/abu-beach-4.csv',
    'data/csv/abu-urban-1.csv',
    'data/csv/abu-urban-2.csv',
    'data/csv/abu-urban-3.csv',
    'data/csv/abu-urban-4.csv',
    'data/csv/abu-urban-5.csv'
]

def import_hsi(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    [x_dim, y_dim, s_dim] = [int(x) for x in lines[0].split(',')]

    truth_map = np.zeros((y_dim, x_dim), dtype='int')
    truth_map_flat = [int(x) for x in lines[s_dim+1].split(',')]
    for y in range(y_dim):
        for x in range(x_dim):
            truth_map[y][x] = truth_map_flat[y*x_dim+x]

    hsi = np.zeros((s_dim, y_dim, x_dim), dtype='double')
    for s, line in enumerate(lines[1:s_dim+1]):
        band_flat = [float(x) for x in line.split(',')]
        for y in range(y_dim):
            for x in range(x_dim):
                hsi[s][y][x] = band_flat[y*x_dim+x]
    
    return [[x_dim, y_dim, s_dim], hsi, truth_map]

def pixel_plot(data, grey=False):
    fig = plt.figure()
    fig.add_axes()
    if grey:
        fig = plt.imshow(data, cmap='Greys_r', interpolation='none')
    else:
        fig = plt.imshow(data, interpolation='none')
    plt.colorbar(fig)
    plt.show(fig)

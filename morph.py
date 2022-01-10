import numpy as np

def erosion(in_image, se_size):
    se_size = se_size // 2
    y_dim, x_dim = np.shape(in_image)
    tmp_image = np.copy(in_image)
    res_image = np.copy(in_image)

    for x in range(x_dim):
        min_index_x = max(0, x-se_size)
        max_index_x = min(x+se_size, x_dim-1)
        for y in range(y_dim):
            tmp_image[y][x] = np.min([in_image[y][j] for j in range(min_index_x, max_index_x+1)])

    for y in range(y_dim):
        min_index_y = max(0, y-se_size)
        max_index_y = min(y+se_size, y_dim-1)
        for x in range(x_dim):
            res_image[y][x] = np.min([tmp_image[i][x] for i in range(min_index_y, max_index_y+1)])

    return res_image

def dilation(in_image, se_size):
    se_size = se_size // 2
    y_dim, x_dim = np.shape(in_image)
    tmp_image = np.copy(in_image)
    res_image = np.copy(in_image)

    for x in range(x_dim):
        min_index_x = max(0, x-se_size)
        max_index_x = min(x+se_size, x_dim-1)
        for y in range(y_dim):
            tmp_image[y][x] = np.max([in_image[y][j] for j in range(min_index_x, max_index_x+1)])

    for y in range(y_dim):
        min_index_y = max(0, y-se_size)
        max_index_y = min(y+se_size, y_dim-1)
        for x in range(x_dim):
            res_image[y][x] = np.max([tmp_image[i][x] for i in range(min_index_y, max_index_y+1)])

    return res_image

def opening(in_image, se_size):
    tmp_image = erosion(in_image, se_size)
    res_image = dilation(tmp_image, se_size)

    return res_image

def closing(in_image, se_size):
    tmp_image = dilation(in_image, se_size)
    res_image = erosion(tmp_image, se_size)

    return res_image

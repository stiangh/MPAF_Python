import numpy as np

"""
def erosion_1(in_image, se_size):
    y_dim, x_dim = np.shape(in_image)
    res_image = np.copy(in_image)

    for y in range(y_dim):
        min_index_y = max(0, y-se_size)
        max_index_y = min(y+se_size, y_dim-1)
        for x in range(x_dim):
            min_index_x = max(0, x-se_size)
            max_index_x = min(x+se_size, x_dim-1)
            res_image[y][x] = np.min([[in_image[i][j] for j in range(min_index_x, max_index_x+1)] for i in range(min_index_y, max_index_y+1)])

    return res_image
"""

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

if __name__ == "__main__":
    # A = np.random.randint(1, 99, (5, 5))
    # B = erosion(A, 1)

    # print(A, B, sep='\n\n')

    from hsi_io import import_hsi, ABU_FILES
    import matplotlib.pyplot as plt

    _, hsi, _ = import_hsi(ABU_FILES[0])

    band = hsi[100]

    eroded = erosion(band, 2)
    dilated = dilation(band, 2)
    opened = opening(band, 2)
    closed = closing(band, 2)

    plt.subplot(2, 3, 1)
    plt.imshow(band, interpolation='none')
    plt.colorbar()
    plt.title('Original')

    plt.subplot(2, 3, 4)
    plt.imshow(band, interpolation='none')
    plt.title('Original')

    plt.subplot(2, 3, 2)
    plt.imshow(eroded, interpolation='none')
    plt.title('Erosion')

    plt.subplot(2, 3, 3)
    plt.imshow(dilated, interpolation='none')
    plt.title('Dilation')

    plt.subplot(2, 3, 5)
    plt.imshow(opened, interpolation='none')
    plt.title('Opening')

    plt.subplot(2, 3, 6)
    plt.imshow(closed, interpolation='none')
    plt.title('Closing')

    plt.show()

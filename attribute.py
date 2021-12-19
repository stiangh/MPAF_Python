import matplotlib
import numpy as np

def tresh(in_image, t, geq=True):
    # geq = True  if pixels greater or equal to threshold should be 1
    # geq = False if pixels lesser  or equal to threshold should be 1
    if geq:
        return in_image >= t
    else:
        return in_image <= t

def connected_components(in_image):
    # in_image is binary image
    x_dim, y_dim = np.shape(in_image)

    id_image = np.zeros((x_dim, y_dim), dtype='int')
    area_image = np.zeros((x_dim, y_dim), dtype='int')

    next_id = 1
    connection_table = {}
    area_table = {}

    # Horizontal lines pass
    for y in range(y_dim):
        for x in range(x_dim):
            if in_image[y][x] == False:
                # Does not fulfill criterion, skip
                continue
            else:
                if (x-1) >= 0 and in_image[y][x-1] == True:
                    # Copy previous ID
                    pixel_id = id_image[y][x-1]
                    id_image[y][x] = pixel_id
                    area_table[pixel_id] += 1
                elif (x+1) < x_dim and in_image[y][x+1] == True:
                    # Create new ID
                    pixel_id = next_id
                    next_id += 1
                    id_image[y][x] = pixel_id
                    area_table[pixel_id] = 1
                    connection_table[pixel_id] = pixel_id

    # Vertical lines pass
    for x in range(x_dim):
        for y in range(y_dim):
            if in_image[y][x] == False:
                # Does not fulfill criterion, skip
                continue
            elif id_image[y][x] == 0:
                # pixel does not have own ID yet
                if (y-1) >= 0 and in_image[y-1][x] == True:
                    # Copy prev id
                    pixel_id = id_image[y-1][x]
                    id_image[y][x] = pixel_id
                    area_table[pixel_id] += 1
                elif (y+1) < y_dim and in_image[y+1][x] == True:
                    # Create new ID
                    pixel_id = next_id
                    next_id += 1
                    id_image[y][x] = pixel_id
                    connection_table[pixel_id] = pixel_id
                    area_table[pixel_id] = 1
            else:
                # Pixel has ID, make connection with prev ID if there is one
                if (y-1) >= 0 and in_image[y-1][x] == True:
                    # prev pixel will have ID, make connection
                    pixel_id = id_image[y][x]
                    prev_pixel_id = id_image[y-1][x]
                    if connection_table[pixel_id] < connection_table[prev_pixel_id]:
                        connection_table[prev_pixel_id] = connection_table[pixel_id]
                    else:
                        connection_table[pixel_id] = connection_table[prev_pixel_id]

    # Collapse areas
    for key_id in connection_table.keys():
        pointer_id = connection_table[key_id]
        if key_id != pointer_id:
            area_table[pointer_id] += area_table[key_id]

    # Update ids and write area image
    for y in range(y_dim):
        for x in range(x_dim):
            if id_image[y][x] > 0:
                pixel_id = id_image[y][x]
                pixel_id = connection_table[pixel_id]
                id_image[y][x] = pixel_id
                area_image[y][x] = area_table[pixel_id]
            
    return id_image, area_image

if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    x_dim, y_dim = 20, 20
    A = np.random.random((x_dim, y_dim))
    B = A >= 0.7

    id_image, area_image = connected_components(B)

    cmap = mpl.cm.get_cmap('viridis')
    cmap.set_under(color='black')

    plt.subplot(1, 3, 1)
    plt.imshow(B, vmin=0.1)
    plt.title('Input')

    plt.subplot(1, 3, 2)
    plt.imshow(id_image, vmin=0.1)
    plt.title('Connected IDs')

    plt.subplot(1, 3, 3)
    plt.imshow(area_image, vmin=0.1)
    plt.title('Connected Area')

    plt.show()

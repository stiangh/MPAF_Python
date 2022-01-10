import numpy as np

def normalize(in_image):
    s = np.std(in_image)
    m = np.mean(in_image)
    res_image = (in_image-m)/(6*s) + 0.5

    y_dim, x_dim = np.shape(in_image)

    for y in range(y_dim):
        for x in range(x_dim):
            val = max(0, res_image[y][x])
            val = min(1, val)
            res_image[y][x] = val

    return res_image

def entropy(in_image):
    bins = [x/100 for x in range(0, 101)]
    hist, _ = np.histogram(in_image.flatten(), bins=bins)
    sum_of_hist = np.sum(hist)
    probs = [h/sum_of_hist for h in hist]

    sum = 0
    for p in probs:
        if p == 0:
            continue
        sum += p * np.log2(p)
    return - sum

def otsu(in_image):
    gray_levels = [x/100 for x in range(101)]
    [hist, _] = np.histogram(in_image.flatten(), bins=gray_levels)
    # Normalization so we have probabilities-like values (sum=1)
    hist = 1.0*hist/np.sum(hist)

    val_max = float('-inf')
    thr = -1
    for t in range(1,101):
        # Non-efficient implementation
        q1 = np.sum(hist[:t])
        q2 = np.sum(hist[t:])
        m1 = np.sum(np.array([i for i in range(t)])*hist[:t])/q1
        m2 = np.sum(np.array([i for i in range(t,100)])*hist[t:])/q2
        val = q1*(1-q1)*np.power(m1-m2,2)
        if val_max < val:
            val_max = val
            thr = t

    return gray_levels[thr]

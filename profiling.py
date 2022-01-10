from hsi_io import ABU_FILES, import_hsi
from mpaf import MPAF

import cProfile
import pstats
import io

# This scipt profiles the algorithm and writes result to output file

t, u, alpha, beta = 10, 4, 0.15, 0.04
se2, se3 = 3, 3

pr = cProfile.Profile()

for i, filename in enumerate(ABU_FILES):
    name = filename[9:-4]

    print(f'name: {name}')

    [x_dim, y_dim, s_dim], hsi, truth = import_hsi(filename)

    pr.enable()
    O = MPAF(hsi, s_dim, t, u, alpha, beta, se2, se3)
    pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open("profile.txt", 'w+') as f:
    f.write(s.getvalue())

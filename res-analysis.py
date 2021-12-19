import numpy as np

best_min = 0
best_min_se2 = None
best_min_se3 = None

best_avg = 0
best_avg_se2 = None
best_avg_se3 = None

best_avg_m2 = 0
best_avg_m2_se2 = None
best_avg_m2_se3 = None

best_avg_m3 = 0
best_avg_m3_se2 = None
best_avg_m3_se3 = None

best_airport = 0
best_airport_se2 = None
best_airport_se3 = None

best_beach = 0
best_beach_se2 = None
best_beach_se3 = None

best_urban = 0
best_urban_se2 = None
best_urban_se3 = None

for se2 in [1, 3, 5, 7]:
    for se3 in [1, 3, 5, 7]:
        filename = f'res/AUC_w_area_sel_{se2}_{se3}.csv'

        with open(filename, 'r') as file:
            lines = file.readlines()

        data = [float(line.strip().split(',')[1]) for line in lines[1:14]]
        data = sorted(data)

        min_val = data[0]
        if min_val > best_min:
            best_min = min_val
            best_min_se2 = se2
            best_min_se3 = se3

        avg = np.mean(data)
        if avg > best_avg:
            best_avg = avg
            best_avg_se2 = se2
            best_avg_se3 = se3

        avg_m2 = np.mean(data[2:])
        if avg_m2 > best_avg_m2:
            best_avg_m2 = avg_m2
            best_avg_m2_se2 = se2
            best_avg_m2_se3 = se3

        avg_m3 = np.mean(data[3:])
        if avg_m3 > best_avg_m3:
            best_avg_m3 = avg_m3
            best_avg_m3_se2 = se2
            best_avg_m3_se3 = se3

        airport_avg = float(lines[-4].strip().split(',')[1])
        beach_avg = float(lines[-3].strip().split(',')[1])
        urban_avg = float(lines[-2].strip().split(',')[1])

        if airport_avg > best_airport:
            best_airport = airport_avg
            best_airport_se2 = se2
            best_airport_se3 = se3

        if beach_avg > best_beach:
            best_beach = beach_avg
            best_beach_se2 = se2
            best_beach_se3 = se3

        if urban_avg > best_urban:
            best_urban = urban_avg
            best_urban_se2 = se2
            best_urban_se3 = se3

print(f'Best min: se2={best_min_se2},se3={best_min_se3}')
print(best_min)
print()
print(f'Best avg: se2={best_avg_se2},se3={best_avg_se3}')
print(best_avg)
print()
print(f'Best avg -m2: se2={best_avg_m2_se2},se3={best_avg_m2_se3}')
print(best_avg_m2)
print()
print(f'Best avg -m3: se2={best_avg_m3_se2},se3={best_avg_m3_se3}')
print(best_avg_m3)
print()
print(f'Best airport: se2={best_airport_se2},se3={best_airport_se3}')
print(best_airport)
print()
print(f'Best beach: se2={best_beach_se2},se3={best_beach_se3}')
print(best_beach)
print()
print(f'Best urban: se2={best_urban_se2},se3={best_urban_se3}')
print(best_urban)
print()
print(f'Best-from-each-scene avg: {np.mean([best_airport, best_beach, best_urban])}')

from glob import glob
import numpy as np
import os

## Output with all the fitted data
## list of volumes in the same order as ev.dat
outputs = open('source.list', 'r').readlines()
outputs = [i.strip() for i in outputs]
prefix = 'fix_distance/cao.thermal-data'

vols, _ = np.loadtxt('ev.dat', unpack=True)
## same temps as for the fitting
temps = np.arange(0, 2750, 10) # temperature list (0 is always included) eg: np.arange(440, 0, -10)
os.system('rm source 2>/dev/null')
os.system('mkdir source 2>/dev/null')
list_gvt = open('./source/list.gvt', 'w')
for i , j in enumerate(temps):
    print(f'{j:10.2f} F_{int(j):04d}.txt', file=list_gvt)
    fout = f'./source/F_{j:04d}.txt'
    write = []
    for fit, vol in zip(outputs, vols):
        file = f'{fit}/{prefix}'
        data = np.loadtxt(f'{fit}{prefix}')
        write.append([vol,  data[i][1], data[i][2], data[i][3]])
    with open(fout, 'w') as ffout:
        for j in write:
            print(*j, file=ffout)



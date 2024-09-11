import numpy as np
import os
from re import search 
from glob import glob

__version__ = '1.0'
__author__ = 'Ernesto Blancas'


'''
This script provides the source folder for the external fvib method implemented
in Gibbs2.

Parameters
----------
ev_filename : str
    Name of the ev list. The first line must be the volume and the last one
    must be the folder of the corresponding volume. For example:
        365.39335000 -152.89797541 .. .. ##vol-10/
        or 
        365.39335000 -152.89797541 .. .. vol-10/
thermal_data_filename : str
    Additional path to the blah.thermal-data file. For example:
        thermal_data_filename = 'max_cutoff/blah.thermal-data'
temperatures : np.array or list
    Array with all the temperatures written using 6-fit_debye_model.py

Returs
------
Nothing

'''
## INPUT
ev_filename = 'ev.dat'
thermal_data_filename = 'max_5.0_3.5_3.0_2.5/casio3.thermal-data'
temperatures = np.arange(0, 2010, 10) # extended temperature list



data = open(ev_filename, 'r').readlines()
data = [i.split() for i in data]
vols = [float(i[0]) for i in data]
outputs = [search('[a-z]+-[0-9]+/', i[-1]).group(0) for i in data]

os.system('rm source 2>/dev/null') ## remove source just in case
os.system('mkdir source 2>/dev/null')
list_gvt = open('./source/list.gvt', 'w')
for i , j in enumerate(temperatures):
    print(f'{j:10.2f} F_{int(j):04d}.txt', file=list_gvt)
    fout = f'./source/F_{j:04d}.txt'
    write = []
    for folder, vol in zip(outputs, vols):
        data = np.loadtxt(f'{folder}/{thermal_data_filename}')
        write.append([vol,  data[i][1], data[i][2], data[i][3]])
    with open(fout, 'w') as fout:
        for j in write:
            print(*j, file=fout)



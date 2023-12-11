from ase.io import read
from pickle import load
from glob import glob
from hiphive.utilities import get_displacements
from numpy import sqrt, loadtxt
import matplotlib.pyplot as plt 

prefix = 'blah'
fout_name = 'disp.dat'                                                          
fout = open(fout_name ,'w') 
print('## D (ang) F(eV/ang) fx fy fz', file=fout)
outputs = 'job*/*.in'

with open(prefix + ".info","rb") as f:
    calculator, phcalc, ncell, cell, scel, fc_factor, phcel = load(f)


for fname in glob(outputs):
    print(fname)
    atoms = read(fname)
    # this is because otherwise the atoms are not in POSCAR order
    displacements = get_displacements(atoms, scel)
    #forces = atoms.get_forces()
    forces = displacements
    ## save displacements and forces to visualize (angs and ev/angs)
    for f, d in zip(forces, displacements):
        print(f'{sqrt(d[0]**2+d[1]**2+d[2]**2):12.4f} {sqrt(f[0]**2+f[1]**2+f[2]**2):12.4f} {f[0]:12.4f} {f[1]:12.4f} {f[2]:12.4f}', file=fout)#, np.abs(f), f, file=fout)

import matplotlib.pyplot as plt 
fout.close()
data = loadtxt(fout_name)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6.4*1, 4.8*1))
ax.hist(data[:,0], 100)
ax.set_ylabel('Counts')
ax.set_xlabel('Amplitude (ang)')
fig.savefig('histogram.pdf')

from ase.io import read
from pickle import load
from glob import glob
from hiphive.utilities import get_displacements
from numpy import sqrt, loadtxt, where, array
import matplotlib.pyplot as plt 

prefix = 'cubic'
outputs = 'harm*/*scf.out'



with open(prefix + ".info","rb") as f:
    [calculator, maximum_cutoff, acoustic_sum_rules, phcalc, ncell, cell, scel, fc_factor, phcel] = load(f)

#labels = set(scel.get_chemical_symbols())
#colors = ['red', 'blue']
#alphas = [1, 0.5]

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6.4*2, 4.8*1))
#for atom, color, alpha in zip(labels, colors, alphas):

for atom in sorted(list(set(scel.get_chemical_symbols()))):
    print(atom)
    fout_name = f'disp-{atom}.dat'                                                          
    fout = open(fout_name ,'w') 
    print('## D (ang) F(eV/ang) fx fy fz', file=fout)
    index = where(array(scel.get_chemical_symbols()) == atom)
    for fname in glob(outputs):
        #print(fname)
        atoms = read(fname)
        # this is because otherwise the atoms are not in POSCAR order
        displacements = get_displacements(atoms, scel)
        displacements = [displacements[i] for i in index[0]]
        forces = atoms.get_forces()
        #forces = displacements
        ## save displacements and forces to visualize (angs and ev/angs)
        for f, d in zip(forces, displacements):
            print(f"""{sqrt(d[0]**2+d[1]**2+d[2]**2):12.4f}\
                  {sqrt(f[0]**2+f[1]**2+f[2]**2):12.4f} {f[0]:12.4f}\
                  {f[1]:12.4f} {f[2]:12.4f}""", file=fout)


    fout.close()
    data = loadtxt(fout_name)
    ax[0].hist(data[:,0], 100, label=atom, alpha=0.7)
    ax[0].set_ylabel('Counts')
    ax[0].legend()
    ax[0].set_xlabel('Amplitude (ang)')
    ax[1].hist(data[:,1], 100, label=atom, alpha=0.7)
    ax[1].set_ylabel('Counts')
    ax[1].legend()
    ax[1].set_xlabel('Amplitude (ang)')
    fig.savefig(f'histogram.pdf')

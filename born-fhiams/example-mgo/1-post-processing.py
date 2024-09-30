#!/home/ernesto/venvs/mod-hip/bin/python3.11
import os, shutil
from pickle import load
import numpy as np
from ase.io import read
from pathlib import Path
""" 
Parameters
"""
##############################################################################
dielectric_output = 'dielectric/born.out'
born_output = 'born.out'
preprocessing_file = 'preprocessing.info'
##############################################################################

def grep_dielectric(filename):
    """ 
    Function to read the FHI-aims output and grep the  dielectric tensor
    Parameters
    ----------
        filename str
        Name of the aims output
    Returns
    -------
        dielectric np.array
        Dielectric tensor
    """
    from re import search
    fin = open(filename, 'r').readlines()
    pattern = 'DFPT for dielectric_constant:*'
    for index, line in enumerate(fin):
        match = search(pattern, line)
        if match:
        #    print(index)
            print('Dielectric tensor found')
            break
    ## lee fin hasta llegar a la line que queremmos
    fin = [i.split() for i in fin]
    tensor = fin[index+1:index+4]
    tensor = [float(j) for i in tensor for j in i]
    return np.array(tensor).reshape((1, 9))

def grep_polarization(filename):
    """ 
    Function to read the FHI-aims output and grep the polarization vector in
    cartesian coordinates
    Parameters
    ----------
    filename str
    Name of the aims output
    """
    fin = open(filename, 'r')
    for line in fin:
        nline = line.split()
        #Cartesian Polarization
        if 'Cartesian' in nline and 'Polarization' in nline:
            polarization = [float(j) for j in  nline[-3:len(nline)]]
            break
    return np.array(polarization)

def calculate_born(cell, atom, index, amplitudes=[0.01, 0.05], fout='aims.out'):
    """
    Calculates the Born effective charges for a given atom
    Parameters
    ----------
        initial_cell ase.Atoms
        Cell with the initial geometry
        atom string
        Symbol of the displaced atom
        index int
        Index of the displaced atom
        amplitude list
        List with the amplitude of the displacements in angstrom
        fout str
        Name of the FHI-aims output file
    Returns
    -------
        born list
        Born effective charges of the corresponding atom 
    """
    dir = ['x', 'y', 'z']
    print(amplitudes)
    volume = cell.get_volume() * 1e-20 ## angstrom**3
    e = 1.6021766e-19
    borns = []
    for i, d in enumerate(dir):
        delta_R = np.abs(amplitudes[0] * np.linalg.norm(cell.cell[i]) - amplitudes[1] * np.linalg.norm(cell.cell[i]))
        p1 = grep_polarization(f'born/{atom}_{index:04d}_{d}_{amplitudes[0]}/{fout}')
        p2 = grep_polarization(f'born/{atom}_{index:04d}_{d}_{amplitudes[1]}/{fout}')
        borns.append(((p2-p1)/delta_R)*(volume/e))
        #print(born*(volume/e))
    born = np.array(borns)
    born[np.where(np.abs(born) <= 1e-6)] = 0
    # print(born.reshape((3,3)))
    return born

def write_BORN(dielectric, born):
    """
    Write the BORN file using Phonopy format and the units of FHI-aims
    Parameters
    ----------
        born list
        Born effective charges of the non-equivalent atoms
        dielectric np.array
        Dielectric tensor
    """
    fout = open('./BORN', 'w')
    factor = 14.399652 ## from phonopy
    print(factor, file=fout)
    die_str = [f'{float(i):12.6f}' for i in dielectric[0]]
    print("".join(die_str), file=fout)
    for i in born:
        born_str = [f'{float(j):12.6f}' for j in i.reshape((1,9))[0]]
        print("".join(born_str), file=fout)

    fout.close()


with open(preprocessing_file, 'rb') as fin:
    [cell_name, cell, unique_index, unique_atoms, amplitudes] = load(fin)

born = []
for i, j in zip(unique_atoms, unique_index):
    print(i, j)
    born.append(calculate_born(cell, i, j, amplitudes, fout=born_output))
dielectric = grep_dielectric(dielectric_output)

write_BORN(dielectric, born)

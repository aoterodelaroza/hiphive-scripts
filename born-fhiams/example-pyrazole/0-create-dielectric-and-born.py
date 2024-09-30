#!/home/ernesto/venvs/mod-hip/bin/python3.11

import os, shutil
import numpy as np
from ase.io import read
from pathlib import Path
from pickle import dump


###################### INPUT BLOCK ############################################
cell_name = 'geometry.in'
amplitudes = [-0.001, 0.001,]
kgrid = [[15, 4, 4], [4, 15, 4], [4, 4, 15]]
###################### INPUT BLOCK ############################################
def main(cell_name, amplitudes, kgrid):
    """ 
    Function to create all the structures need for the calculation of the Born
    effective charges and the dielectric tensor. If found its also copies the
    control.in into each folder
    Parameters
    ----------
        cell_name string
        Name of the file with the structure
        amplitudes list
        List with the amplitudes for the calculation of the born effective charges
        kgrid list
        List of list for the calculation of the polarization along x, y and z
    Returns
    -------
    """
    cell = read(cell_name)
    unique_index, unique_atoms = find_unique_atoms(cell)
    for i, j in zip(unique_index, unique_atoms):
        print(i, j)
        displace(cell, j, i, kgrid, amplitudes)
    dielectric(cell_name)
    with open('preprocessing.info', 'wb') as fout:
        dump([cell_name, cell, unique_index, unique_atoms, amplitudes], fout)


def find_unique_atoms(cell):
    """
    Function to find all the non-equivalent atoms in the cell
    Parameters
    ----------
        cell ase.Atoms
        Cell use for the calculation
    Returns
    -------
        unique_index np.array 
        List with the index of the non equivalent atoms
        unique_atoms list
        List with the symbols of the non equivalent atoms
    """

    import spglib as spg
    spg_cell = (cell.cell, cell.get_scaled_positions(),
                cell.get_atomic_numbers())
    sym = spg.get_symmetry_dataset(spg_cell, symprec=1e-5)
    unique = sym['equivalent_atoms']
    unique_index = np.unique(unique)
    unique_atoms = [cell.get_chemical_symbols()[i] for i in unique_index]

    return unique_index, unique_atoms

def displace(initial_cell, atom, index, k_grid, amplitude=[-0.01, 0.01]):
    """
    Function to create all the displacements necessary for the Born effective
    charges calculation
    Parameters
    ----------
        initial_cell ase.Atoms
        Cell use for the calculation
        atom string
        Symbol of the displaced atom
        index int
        Index of the displaced atom
        k_grid list
        k-point grid for the calculation of the charges
        amplitude list
        List with the amplitude of the displacements in angstrom
    Returns
    -------
    Create all the folders and the geometry inputs 
    """
    dir = ['x', 'y', 'z']
    Path('./born').mkdir(parents=True, exist_ok=True)
    for a in amplitude:
        for i, d in enumerate(dir):
            folder = f'born/{atom}_{index:04d}_{d}_{a}'
            Path(folder).mkdir(parents=True, exist_ok=True)
            new = initial_cell.copy()
            new.positions[index][i] += a* np.linalg.norm(new.cell[i])
            new.write(f'{folder}/geometry.in')
            if os.path.isfile('control.in'):
                #shutil.copy('control.in', f'{folder}/control.in')
                control = open('control.in', 'r')
                fout = open(f'{folder}/control.in', 'w')
                for line in control.readlines():
                    print(line.strip(), file=fout)
                    if 'k_grid' in line:
                        for ik, kpath in enumerate(k_grid):
                            print(f'output polarization {ik+1} {kpath[0]} {kpath[1]} {kpath[2]}', 
                                  file=fout)
                control.close()
                fout.close()

def dielectric(cell_name):
    """
    Function to create the directory for the dielectric constants calculation
    Parameters
    ----------
        cell_name string
        String with the name of the input geometry
    Returns
    -------
    Create all the folders and the geometry inputs 
    """

    Path('./dielectric').mkdir(parents=True, exist_ok=True)
    shutil.copy(f'{cell_name}', 'dielectric/geometry.in')
    if os.path.isfile('control.in'):
        control = open('control.in', 'r')
        fout = open('dielectric/control.in', 'w')
        for line in control.readlines():
            print(line.strip(), file=fout)
            if 'k_grid' in line:
                print('DFPT dielectric', file=fout)
        control.close()
        fout.close()


main(cell_name, amplitudes, kgrid)

## 3-generate_structures.py: generate random structures by phonon rattling.
##
## Input: prefix.info
## Output: a number of input structures in prefix*/

## input block ##
prefix="mgo" ## prefix for the generated files
fc2_phonopy = None ## if given, read the FC2s from phonopy (FORCE_CONSTANTS file)
rattle = [(500, 20)] ## rattle type: list of (T,nstruct)
out_kwargs = {
    'prefix': 'crystal',
    'pseudo_dir': '../..',
    'tprnfor': True,
    'ecutwfc': 80.0,
    'ecutrho': 800.0,
    'calculation': 'scf',
    'conv_thr': 1e-10,
    'pseudopotentials': {'Sr': 'sr.UPF', 'Ti': 'ti.UPF', 'O': 'o.UPF'},
    'kpts': (3, 3, 3),
} ## pass this down to ASE (example for QE)
## out_kwargs = {} ## pass this down to ASE (example for VASP,FHIaims)
#################

import os
import pickle
import ase.io
import hiphive as hp
from hiphive.structure_generation import generate_phonon_rattled_structures

# create the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, phcalc, ncell, cell, scel, fc_factor, phcel = pickle.load(f)

# 2nd-order force constant (text-mode), need to be in eV/ang**2
if (fc2_phonopy):
    fc2 = hp.ForceConstants.read_phonopy(supercell=scel,fname=fc2_phonopy,format='text')
    fc2 = fc2.get_fc_array(order=2,format='ase')
    fc2 = fc2 * fc_factor

else:
    with open(prefix + ".fc2_harmonic","rb") as f:
        fc2 = pickle.load(f)
        fc2 = fc2 * fc_factor

# generate the structures
for rr in rattle:
    structures = generate_phonon_rattled_structures(scel,fc2,rr[1],rr[0])
    for iz in enumerate(structures):
        dirname = f'{prefix}-{rr[0]:04d}-{iz[0]:02d}'
        if os.path.isdir(dirname):
            raise Exception("directory " + dirname + " already exists")
        os.mkdir(dirname)
        if calculator == "vasp":
            filename = f"{dirname}/POSCAR"
        elif calculator == "espresso-in":
            filename = f'{dirname}/{dirname}.scf.in'
        elif calculator == "aims":
            filename = f'{dirname}/geometry.in'
        ase.io.write(filename,iz[1],format=calculator,**out_kwargs)

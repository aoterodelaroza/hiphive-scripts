## 1a-generate_structures.py: generate random structures by phonon rattling.
##
## Input: prefix.info
## Output: a number of input structures in prefix*/

## input block ##
prefix="blah" ## prefix for the generated files
fc2_phonopy = None ## if given, read the FC2s from phonopy (FORCE_CONSTANTS)
rattle = [(1000,10)] ## rattle type: list of (T,nstruct)
out_kwargs = {
    'prefix': 'crystal',
    'pseudo_dir': '../..',
    'tprnfor': True,
    'ecutwfc': 80.0,
    'ecutrho': 800.0,
    'conv_thr': 1e-10,
    'pseudopotentials': {'O': 'o.UPF', 'Mg': 'mg.UPF'},
    'kpts': (3,3,3),
} ## pass this down to ASE (example for QE)
#################

import os
import pickle
import ase.io
import hiphive as hp
from hiphive.structure_generation import generate_phonon_rattled_structures

# create the info file
with open(prefix + ".info","rb") as f:
    calculator, phcalc, ncell, cell, scel = pickle.load(f)

# 2nd-order force constant (text-mode)
if (fc2_phonopy):
    fc2 = hp.ForceConstants.read_phonopy(supercell=scel,fname=fc2,format='text')
    fc2 = fc2.get_fc_array(order=2,format='ase')
else:
    with open(prefix + ".fc2_harmonic","rb") as f:
        fc2 = pickle.load(f)

# generate the structures
for rr in rattle:
    structures = generate_phonon_rattled_structures(scel,fc2,rr[1],rr[0])
    for iz in enumerate(structures):
        name = prefix + "-%d-%d" % (rr[0],iz[0])
        if os.path.isdir(name):
            raise Exception("directory " + name + " already exists")
        os.mkdir(name)
        if calculator == "vasp":
            filename = name + "/POSCAR"
        elif calculator == "espresso-in":
            filename = name + "/" + name + ".scf.in"
        ase.io.write(filename,iz[1],format=calculator,**out_kwargs)

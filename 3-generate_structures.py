## 1a-generate_structures.py: generate random structures by phonon rattling.
##
## Input: prefix.info
## Output: a number of input structures in prefix*/

## input block ##
prefix="blah" ## prefix for the generated files
fc2 = "FORCE_CONSTANTS_2ND" ## FC2s from phonopy/elsewhere (text)
rattle = [(300,12),(500,12)] ## rattle type: list of (T,nstruct)
out_kwargs = {} ## pass this down to ASE
#################

import os
import pickle
import ase.io
import hiphive as hp
from hiphive.structure_generation import generate_phonon_rattled_structures

# create the info file
with open(prefix + ".info","rb") as f:
    calculator, ncell, cell, scel = pickle.load(f)

# 2nd-order force constant (text-mode)
fc2 = hp.ForceConstants.read_phonopy(supercell=scel,fname=fc2,format='text')
fc2 = fc2.get_fc_array(order=2,format='ase')

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
        else:
            filename = name + "/" + name
        ase.io.write(filename,iz[1],format=calculator,**out_kwargs)

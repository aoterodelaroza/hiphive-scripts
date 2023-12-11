## 1a-generate_structures.py: generate random structures by phonon rattling.
##
## Input: prefix.info
## Output: a number of input structures in prefix*/

## input block ##
prefix="blah" ## prefix for the generated files
fc2_phonopy = None ## if given, read the FC2s from phonopy (FORCE_CONSTANTS)
rattle = [(500, 20)] ## rattle type: list of (T,nstruct)
name = 'job' ## folder name 

out_kwargs = {
    'prefix': 'crystal',
    'pseudo_dir': '../../pseudo',
    'tprnfor': True,
    'ecutwfc': 60.0,
    'ecutrho': 600.0,
    'calculation': 'scf',
    'tprnfor': True,
    'tstress': True,
    'etot_conv_thr': 1e-5,
    'forc_conv_thr': 1e-4,
    'conv_thr': 1e-10,
    'pseudopotentials': {'Sr': 'sr.UPF', 'Ti': 'ti.UPF', 'O': 'o.UPF'},
    'kpts': (3, 3, 3),
} ## pass this down to ASE (example for QE)
#################

import os
import pickle
import ase.io
import hiphive as hp
from hiphive.structure_generation import generate_phonon_rattled_structures

# create the info file
with open(prefix + ".info","rb") as f:
    ## EB units added
    calculator, phcalc, ncell, cell, scel, fc_factor, phcel = pickle.load(f)

# 2nd-order force constant (text-mode)
## EB FC2 need to be in eV/ang**2
if (fc2_phonopy):
    fc2 = hp.ForceConstants.read_phonopy(supercell=scel,fname=fc2,format='text')
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
        ## EB
        dirname = f'{name}-{rr[0]:04d}-{iz[0]:02d}'
        print(dirname)
        if os.path.isdir(name):
            raise Exception("directory " + dirname + " already exists")
        os.mkdir(dirname)
        if calculator == "vasp":
            filename = f"{dirname}/POSCAR"
        elif calculator == "espresso-in":
            filename = f'{dirname}/{prefix}.scf.in'
        ase.io.write(filename,iz[1],format=calculator,**out_kwargs)

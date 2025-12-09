## 3-generate_structures.py: generate random structures by phonon rattling.
##
## Input: prefix.info, prefix.fc2_harmonic, prefix.cs_harmonic, prefix.cs
## Output: a number of input structures: anharmonic*/

## input block ##
prefix="urea" ## prefix for the generated files
fc2_phonopy = None ## if given, read the FC2s from phonopy (FORCE_CONSTANTS file)
rattle_temperature = 50 ## rattle temperature
#################

import os
import pickle
import ase
import hiphive as hp
from hiphive.utilities import get_displacements
from hiphive_utilities import generate_phonon_rattled_structures
import numpy as np

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# create the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, nthread_batch_lsqr, phcalc, ncell, cell, cell_for_cs, scel, fc_factor, phcel, out_kwargs, symprec = pickle.load(f)

# 2nd-order force constant (text-mode), need to be in eV/ang**2
if (fc2_phonopy):
    fc2 = hp.ForceConstants.read_phonopy(supercell=scel,fname=fc2_phonopy,format='text')
    fc2 = fc2.get_fc_array(order=2,format='ase')
    fc2 = fc2 * fc_factor
else:
    with open(prefix + ".fc2_harmonic","rb") as f:
        fc2 = pickle.load(f)
        fc2 = fc2 * fc_factor

# read the harmonic and complete cluster spaces
with open(prefix + ".cs_harmonic","rb") as f:
    cutoffs_harmonic,cs_harmonic = pickle.load(f)
with open(prefix + ".cs","rb") as f:
    cutoffs,cs = pickle.load(f)

## calculate the number of structures
nparam = cs.n_dofs - cs_harmonic.n_dofs
n_structures = np.ceil(nparam * 10.0 / (3 * len(scel))).astype(int)
structures = generate_phonon_rattled_structures(scel,fc2,n_structures,rattle_temperature)

# write the structures
print("#id filename avg-displacement",flush=True)
for i, s in enumerate(structures):
    dirname = "anharmonic-%d-%3.3d" % (rattle_temperature,i)
    if os.path.isdir(dirname):
        raise Exception("directory " + dirname + " already exists")
    os.mkdir(dirname)
    if calculator == "vasp":
        filename = f"{dirname}/POSCAR"
    elif calculator == "espresso-in":
        filename = f'{dirname}/{dirname}.scf.in'
    elif calculator == "aims":
        filename = f'{dirname}/geometry.in'
    ase.io.write(filename,s,format=calculator,**out_kwargs)

    print("%4d %s %.5f" % (i,filename,np.mean([np.linalg.norm(d) for d in get_displacements(s,scel)])),flush=True)

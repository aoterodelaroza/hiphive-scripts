## 1a-generate_structures_rattle.py: generate random structures by
## simple structure rattling using the default phonopy
## displacements. These will be used to calculate the first harmonic
## FC2 for the subsequent phonon rattle.
##
## Input: prefix.info
## Output: a number of subdirectories containing the rattled structures

## input block ##
prefix="urea" ## prefix for the generated files
#################

import math
import pickle
import os
import ase, ase.io
import time
import numpy as np
from phonopy.interface.calculator import get_default_displacement_distance, get_default_physical_units
from hiphive_utilities import constant_rattle

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# read the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, nthread_batch_lsqr, phcalc, ncell, cell, cell_for_cs, scel, fc_factor, phcel, out_kwargs = pickle.load(f)
units = get_default_physical_units(phcalc)

# initialize random seed
seed = int(time.time())
print("# Initialize random seed = %d" % seed,flush=True)
rs = np.random.RandomState(seed)

# read the harmonic cluster space
with open(prefix + ".cs_harmonic","rb") as f:
    cutoffs_harmonic,cs_harmonic = pickle.load(f)

## The number of structures is: enough to have at least 10 times as many forces as parameters (rounded up)
n_structures = np.ceil(cs_harmonic.n_dofs * 10.0 / (3 * len(scel))).astype(int)

## Generate rattled structures
rattle_std = get_default_displacement_distance(phcalc) * units["distance_to_A"]
structures = constant_rattle(scel, n_structures, rattle_std, rs)

for iz in enumerate(structures):
    name = 'harmonic' + "-%03d" % iz[0]
    if os.path.isdir(name):
        raise Exception("directory " + name + " already exists")
    os.mkdir(name)
    if calculator == "vasp":
        filename = name + "/POSCAR"
    elif calculator == "espresso-in":
        filename = name + "/" + name + ".scf.in"
    elif calculator == "aims":
        filename = name + "/geometry.in"
    ase.io.write(filename,iz[1],format=calculator,**out_kwargs)

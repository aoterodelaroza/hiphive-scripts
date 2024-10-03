## 1a-generate_structures_rattle.py: generate random structures by
## simple structure rattling using the default phonopy
## displacements. These will be used to calculate the first harmonic
## FC2 for the subsequent phonon rattle.
##
## Input: prefix.info
## Output: a number of subdirectories containing the rattled structures

## input block ##
prefix="blah" ## prefix for the generated files
out_kwargs = {
    'prefix': 'crystal',
    'pseudo_dir': '../..',
    'tprnfor': True,
    'ecutwfc': 80.0,
    'ecutrho': 800.0,
    'conv_thr': 1e-10,
    'pseudopotentials': {'O': 'o.UPF', 'Mg': 'mg.UPF'},
    'kpts': (2,2,2),
} ## pass this down to ASE (example for QE)
## out_kwargs = {} ## pass this down to ASE (example for VASP/FHIaims)
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
    calculator, maximum_cutoff, acoustic_sum_rules, use_batch_lsqr, phcalc, ncell, cell, cell_for_cs, scel, fc_factor, phcel = pickle.load(f)
units = get_default_physical_units(phcalc)

# initialize random seed
seed = int(time.time())
print("# Initialize random seed = %d" % seed)
rs = np.random.RandomState(seed)

## Get the displacement distance and get the number of structures created by phonopy
phdist = get_default_displacement_distance(phcalc)
phcel.generate_displacements(distance=phdist)
n_structures = max(math.ceil(len(phcel.supercells_with_displacements)/3),1)

## Generate rattled structures
rattle_std = phdist * units["distance_to_A"]
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

## 1b-build_cluster_config.py: generate cluster spaces for a given structure
##
## Input: prefix.info
## Output: prefix.cs

## input block ##
prefix="blah" ## prefix for the generated files
cutoffs = [6.32, 3.00] # list of cutoffs [2nd,3rd,...] in angstrom
#################

import pickle
from hiphive import ClusterSpace
import numpy as np
import ase
import phonopy
from phonopy.interface.calculator import get_default_physical_units

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the info file
cutoffs[0] = np.trunc(cutoffs[0]*100)/100 ## truncate to the 2nd decimal place
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, phcalc, ncell, cell, scel, fc_factor, phcel = pickle.load(f)

# build the cluster and save
units = get_default_physical_units(phcalc)
ph = phcel.supercell
cell1 = ase.Atoms(cell=ph.cell*units["distance_to_A"], symbols=ph.symbols,
                  scaled_positions=ph.scaled_positions, pbc=True)
cs = ClusterSpace(cell1, cutoffs, acoustic_sum_rules=acoustic_sum_rules)
with open(prefix + ".cs","wb") as f:
    pickle.dump([cutoffs,cs],f)

# print out some details
print("--- cluster space details ---")
print(cs)
print("")

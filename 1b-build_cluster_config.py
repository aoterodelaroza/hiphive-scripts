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

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the info file
cutoffs[0] = np.trunc(cutoffs[0]*100)/100 ## truncate to the 2nd decimal place
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, phcalc, ncell, cell, cell_for_cs, scel, fc_factor, phcel = pickle.load(f)

# build the cluster and save
cs = ClusterSpace(cell_for_cs, cutoffs, acoustic_sum_rules=acoustic_sum_rules)
with open(prefix + ".cs","wb") as f:
    pickle.dump([cutoffs,cs],f)

# print out some details
print("--- cluster space details ---")
print(cs)
print("")

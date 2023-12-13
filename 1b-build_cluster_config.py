## 1b-build_cluster_config.py: generate cluster spaces for a given structure
##
## Input: prefix.info
## Output: prefix.cs

## input block ##
prefix="blah" ## prefix for the generated files
cutoffs = [6.32, 5.91] # list of cutoffs [2nd,3rd,...] in angstrom
#################

import pickle
from hiphive import ClusterSpace
import numpy as np

# load the info file
cutoffs[0] = np.trunc(cutoffs[0]*100)/100 ## truncate to the 2nd decimal place
with open(prefix + ".info","rb") as f:
    calculator, phcalc, ncell, cell, scel, fc_factor, phcel = pickle.load(f)

# build the cluster and save
cs = ClusterSpace(cell, cutoffs)
with open(prefix + ".cs","wb") as f:
    pickle.dump([cutoffs,cs],f)

# print out some details
print("--- cluster space details ---")
print(cs)
print("")

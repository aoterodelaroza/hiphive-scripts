## 1b-build_cluster_config.py: generate cluster spaces for a given structure
##
## Input: prefix.info
## Output: prefix.cs

## input block ##
prefix="blah" ## prefix for the generated files
## EB cutoffs must be in ang
## octava y septima -0.05 shell 
cutoffs = [6.32, 5.91] # list of cutoffs [2nd,3rd,...] in angstrom
#################

import pickle
from hiphive import ClusterSpace

# load the info file
with open(prefix + ".info","rb") as f:
    ## EB units added 
    calculator, phcalc, ncell, cell, scel, fc_factor, phcel = pickle.load(f)

# build the cluster and save
cs = ClusterSpace(cell, cutoffs)
with open(prefix + ".cs","wb") as f:
    pickle.dump([cutoffs,cs],f)

# print out some details
print("--- cluster space details ---")
print(cs)
print("")

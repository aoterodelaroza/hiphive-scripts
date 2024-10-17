## 1-build_cluster_config.py: generate the harmonic and nth-order
## cluster spaces for a given structure.
##
## Input: prefix.info
## Output: prefix.cs_harmonic prefix.cs

## input block ##
prefix="urea" ## prefix for the generated files
cutoffs = [6.0,4.0,3.0,2.0] # cutoffs for [2nd,3rd,...] order in angstrom
#################

import pickle
from hiphive import ClusterSpace
import numpy as np
import ase

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, nthread_batch_lsqr, phcalc, ncell, cell, cell_for_cs, scel, fc_factor, phcel, out_kwargs = pickle.load(f)

# build the harmonic clusterspace and save
cs_harmonic = ClusterSpace(cell_for_cs, [cutoffs[0]], acoustic_sum_rules=True)
with open(prefix + ".cs_harmonic","wb") as f:
    pickle.dump([[cutoffs[0]],cs_harmonic],f)

# build the complete clusterspace and save
cs = ClusterSpace(cell_for_cs, cutoffs, acoustic_sum_rules=acoustic_sum_rules)
with open(prefix + ".cs","wb") as f:
    pickle.dump([cutoffs,cs],f)

# print out some details
print("--- cluster space details (harmonic) ---",flush=True)
print(cs_harmonic,flush=True)
print("",flush=True)
print("--- cluster space details (complete) ---",flush=True)
print(cs,flush=True)
print("",flush=True)

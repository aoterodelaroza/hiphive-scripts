## 1c-calculate_lr_forces.py: calculate the LR FC2 from the Born charge file.
##
## Input: prefix.info
## Output: prefix.fc2_lr

## input block ##
prefix="blah" ## prefix for the generated files
born_file="BORN" ## file containing the Born charges in VASP format
#################

import pickle
import numpy as np
# from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import parse_BORN
# from phonopy import Phonopy

# load the info file
with open(prefix + ".info","rb") as f:
    ## EB fc_factor added
    calculator, phcalc, ncell, cell, scel, fc_factor, phcel = pickle.load(f)

## EB si habia un born cuando se creo el 0 ya estan los born incluidos
phcel.nac_params = parse_BORN(phcel.primitive, symprec=1e-5, is_symmetry=True,
                              filename='BORN')

## EB update phonopy
## ph.set_force_constants(np.zeros((len(scel), len(scel), 3, 3)))
phcel.force_constants = np.zeros((len(scel), len(scel), 3, 3))
## EB get_dynamical matrix is depreciated
# dynmat = ph.get_dynamical_matrix()
dynmat = phcel.dynamical_matrix
dynmat.make_Gonze_nac_dataset()
#fc2_LR = -dynmat.get_Gonze_nac_dataset()[0]
fc2_LR = -dynmat.Gonze_nac_dataset[0]
##  EB to remove
print(fc2_LR[0][0][0]*fc_factor)
## LR FC2 are written in the calculator units!! 
with open(prefix + ".fc2_lr","wb") as f:
    pickle.dump(fc2_LR, f)
    #pickle.dump(fc2_LR, f)

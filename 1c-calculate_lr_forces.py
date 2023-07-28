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
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import parse_BORN

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, phcalc, ncell, cell, scel = pickle.load(f)

atoms_phonopy = PhonopyAtoms(symbols=cell.get_chemical_symbols(),
                             scaled_positions=cell.get_scaled_positions(),
                             cell=cell.cell)
ph = Phonopy(atoms_phonopy, supercell_matrix=ncell*np.eye(3),
             primitive_matrix=None)

ph.nac_params = parse_BORN(ph.primitive, filename=born_file)
ph.set_force_constants(np.zeros((len(atoms_phonopy), len(atoms_phonopy), 3, 3)))
dynmat = ph.get_dynamical_matrix()
dynmat.make_Gonze_nac_dataset()
fc2_LR = -dynmat.get_Gonze_nac_dataset()[0]

with open(prefix + ".fc2_lr","wb") as f:
    pickle.dump(fc2_LR,f)


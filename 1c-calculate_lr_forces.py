## 1c-calculate_lr_forces.py: calculate the LR FC2 from the Born charge file.
##
## Input: prefix.info
## Output: prefix.fc2_lr

## input block ##
prefix="blah" ## prefix for the generated files
born_file="BORN" ## file containing the Born charges
#################

import pickle
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import parse_BORN
from phonopy.interface.calculator import get_default_physical_units

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, ncell, cell, scel = pickle.load(f)

atoms_phonopy = PhonopyAtoms(symbols=cell.get_chemical_symbols(),
                             scaled_positions=cell.get_scaled_positions(),
                             cell=cell.cell)
ph = Phonopy(atoms_phonopy, supercell_matrix=ncell*np.eye(3),
             primitive_matrix=None,calculator=calculator)

units = get_default_physical_units(calculator)
ph.nac_params = parse_BORN(ph.primitive, filename=born_file)
ph.nac_params['factor'] = units['nac_factor']
ph.set_force_constants(np.zeros((len(atoms_phonopy), len(atoms_phonopy), 3, 3)))
dynmat = ph.get_dynamical_matrix()
dynmat.make_Gonze_nac_dataset()
fc2_LR = -dynmat.get_Gonze_nac_dataset()[0]

with open(prefix + ".fc2_lr","wb") as f:
    pickle.dump(fc2_LR,f)

## with open(prefix + ".fc2_lr","wb") as f:
##     pickle.dump(fc2_LR,f)
##     born = get_born_vasprunxml(filename='./CALC_LR/vasprun.xml',
##                                symprec=1e-5, symmetrize_tensors=True)
##     with open('./CALC_LR/BORN', 'w') as output:
##         # DEFAULT CONVERSION FACTOR
##         print('14.399652', file=output)
##         # DIELECTRIC CONSTANT
##         print(*[i for i in born[1].reshape(1, 9)[0]], file=output)
##         # BORN CHARGES
##         for atom in born[0]:
##             print(*[i for i in atom.reshape(1, 9)[0]], file=output)

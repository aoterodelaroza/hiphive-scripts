## 0-write_structure_info.py: reads the equilibrum structure and
## supercell size, builds the supercell, and saves the information to
## an info file. If the BORN file is available, reads the Born charges
## and creates the second-order force-constant matrix in file fc2_LR.
##
## Input: none
## Output: prefix.info
## --> optional: prefix.fc2_LR

## input block ##
prefix="blah" ## prefix for the generated files
eq_structure="mgo.scf.in" ## the equilibrium structure
ncell = (3 ,3 ,3) ## the supercell size
calculator = "espresso-in" ## program used for the calculations, case insensitive (vasp,espresso-in)
#################

import os
import pickle
import ase
import ase.io
import ase.build
import phonopy
import numpy as np
from phonopy.interface.calculator import get_default_physical_units, get_force_constant_conversion_factor
from hiphive.cutoffs import estimate_maximum_cutoff
from phonopy.file_IO import parse_BORN

# process the calculator
calculator = calculator.lower()
if (calculator == "espresso-in"):
    phcalc = "qe"
elif (calculator == "vasp"):
    phcalc = calculator
else:
    raise Exception("unknown calculator: " + calculator)

# reference cell
cell = ase.io.read(eq_structure)

# supercell: make one phonopy and VASP like
units = get_default_physical_units(phcalc)
ph = phonopy.load(unitcell_filename=eq_structure,supercell_matrix=list(ncell),calculator=phcalc)
phcel = ph ## save the phonopy cell (problems with primitive cells)
ph = ph.supercell
scel = ase.Atoms(symbols=ph.symbols,scaled_positions=ph.scaled_positions,cell=ph.cell*units["distance_to_A"],pbc=[1,1,1])

# if BORN file exists, read the NAC parameters
if os.path.isfile("BORN"):
    print("BORN file is used, generating " + prefix + ".fc2_lr")
    phcel.nac_params = parse_BORN(phcel.primitive, symprec=1e-5, is_symmetry=True,
                                  filename='BORN')
    phcel.force_constants = np.zeros((len(scel), len(scel), 3, 3))
    dynmat = phcel.dynamical_matrix
    dynmat.make_Gonze_nac_dataset()
    fc2_LR = -dynmat.Gonze_nac_dataset[0]

    ## LR FC2 are written in the calculator units!!
    with open(prefix + ".fc2_lr","wb") as f:
        pickle.dump(fc2_LR, f)

# write the max cutoff to output
print("maximum cutoff (angstrom): ",estimate_maximum_cutoff(scel))
fc_factor = get_force_constant_conversion_factor(units['force_constants_unit'],interface_mode='vasp')
print(f'FC factor to eV/ang**2: {fc_factor}')
# create the info file
with open(prefix + ".info","wb") as f:
    pickle.dump([calculator.lower(), phcalc, ncell, cell, scel, fc_factor,
                 phcel],f)

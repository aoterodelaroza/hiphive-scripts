## 0-write_structure_info.py: reads the equilibrum structure and
## supercell size, builds the supercell, and saves the information to
## an info file.
##
## Input: none
## Output: prefix.info

## EB phonopy units need to be on top
## EB all conversion factors imported
from phonopy.units import *
## input block ##
prefix="blah" ## prefix for the generated files
eq_structure="mgo.scf.in" ## the equilibrium structure
ncell = (3 ,3 ,3) ## the supercell size
calculator = "espresso-in" ## program used for the calculations, case insensitive (vasp,espresso-in)
#################

import pickle
import ase
import ase.io
import ase.build
import phonopy
from phonopy.interface.calculator import get_default_physical_units 
from phonopy.interface.calculator import get_force_constant_conversion_factor
from hiphive.cutoffs import estimate_maximum_cutoff

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
## EB new para evitar los problemas de la primitiva
phcel = ph
ph = ph.supercell
scel = ase.Atoms(symbols=ph.symbols,scaled_positions=ph.scaled_positions,cell=ph.cell*units["distance_to_A"],pbc=[1,1,1])

# write the max cutoff to output
print("maximum cutoff (angstrom): ",estimate_maximum_cutoff(scel))
fc_factor = get_force_constant_conversion_factor(units['force_constants_unit'],
        interface_mode='vasp')
print(f'FC factor to eV/ang**2: {fc_factor}')
# create the info file
with open(prefix + ".info","wb") as f:
    pickle.dump([calculator.lower(), phcalc, ncell, cell, scel, fc_factor,
                 phcel],f)


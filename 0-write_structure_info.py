## 0-write_structure_info.py: reads the equilibrum structure and
## supercell size, builds the supercell, and saves the information to
## an info file.
##
## Input: none
## Output: prefix.info

## input block ##
prefix="blah" ## prefix for the generated files
eq_structure="mgo.scf.in" ## the equilibrium structure
ncell = (3,3,3) ## the supercell size
calculator = "espresso-in" ## program used for the calculations, case insensitive (vasp,espresso-in)
#################

import pickle
import ase
import ase.io
import ase.build
import phonopy

# process the calculator
calculator = calculator.lower()
if (calculator == "espresso-in"):
    phcalc = "qe"
else:
    phcalc = calculator

# reference cell
cell = ase.io.read(eq_structure)

# supercell: make one phonopy and VASP like
ph = phonopy.load(unitcell_filename=eq_structure,supercell_matrix=list(ncell),calculator=phcalc)
ph = ph.supercell
scel = ase.Atoms(symbols=ph.symbols,scaled_positions=ph.scaled_positions,cell=ph.cell,pbc=[1,1,1])

# Write a temporary file with the supercell and read it back. If VASP,
# this gets the atoms in POSCAR order.
ase.io.write(prefix + ".tmp_supercell",scel,format=calculator)

# create the info file
with open(prefix + ".info","wb") as f:
    pickle.dump([calculator.lower(),ncell,cell,scel],f)


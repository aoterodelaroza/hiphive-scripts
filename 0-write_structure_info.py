## 0-write_structure_info.py: reads the equilibrum structure and
## supercell size, builds the supercell, and saves this and other
## information to an info file. If the BORN file is available, reads
## the Born charges and creates the second-order force-constant matrix
## in file fc2_LR.
##
## Input: none
## Output: prefix.info
## --> optional: prefix.fc2_LR

## input block ##
prefix="urea" ## prefix for the generated files
eq_structure="urea.orig.in" ## the equilibrium structure
ncell = [1,2,0,-2,1,-1,0,0,3] ## nice supercell
calculator = "espresso-in" ## program used for the calculations, case insensitive (vasp,espresso-in,aims)
maximum_cutoff = 6.2 ## maximum cutoff for this crystal (angstrom, NEWCELL NICE 1 on supercell)
acoustic_sum_rules = False # whether to use acoustic sum rules (fewer parameters, much slower)
nthread_batch_lsqr = 30 # if > 0, use batch least squares (less memory, more CPU) with these many threads
out_kwargs = { ## pass this down to ASE (example for QE)
    'prefix': 'crystal',
    'pseudo_dir': '../',
    'tprnfor': True,
    'ecutwfc': 80.0,
    'ecutrho': 800.0,
    'conv_thr': 1e-10,
    'pseudopotentials': {'C':'c.UPF','O':'o.UPF','N':'n.UPF','H':'h.UPF'},
    'kpts': (2,2,2),
} ## pass this down to ASE (example for QE)
## out_kwargs = {} ## pass this down to ASE (example for VASP/FHIaims)
#################

import os
import pickle
import ase, ase.io
import phonopy
import numpy as np
from phonopy.interface.calculator import get_default_physical_units, get_force_constant_conversion_factor
from phonopy.file_IO import parse_BORN

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

## working with nice supercell
ncell = np.array(ncell)
ncell = ncell.reshape((3, 3))

# process the calculator
calculator = calculator.lower()
if (calculator == "espresso-in"):
    phcalc = "qe"
elif (calculator == "vasp" or calculator == "aims"):
    phcalc = calculator
else:
    raise Exception("unknown calculator: " + calculator)

# reference cell
cell = ase.io.read(eq_structure)

# supercell: make one phonopy and VASP like
units = get_default_physical_units(phcalc)
ph = phonopy.load(unitcell_filename=eq_structure,supercell_matrix=ncell.T,primitive_matrix=np.eye(3),calculator=phcalc,produce_fc=False)
if os.path.isfile("FORCE_CONSTANTS") or os.path.isfile("FORCE_SETS") or os.path.isfile("force_constants.hdf5"):
    raise Exception("FORCE_CONSTANTS/FORCE_SETS/force_constants.hdf5 is present in this directory; stopping")
phcel = ph ## save the phonopy cell (problems with primitive cells)
ph = ph.supercell
scel = ase.Atoms(symbols=ph.symbols,scaled_positions=ph.scaled_positions,cell=ph.cell*units["distance_to_A"],pbc=[1,1,1])

# cell for clusterspace
cell_for_cs = ase.Atoms(cell=ph.cell*units["distance_to_A"], symbols=ph.symbols,
                        scaled_positions=ph.scaled_positions, pbc=True)

## additional to check if supercell is ok
ase.io.write('supercell.geometry.in',scel,format="aims")

# if BORN file exists, read the NAC parameters
if os.path.isfile("BORN"):
    print("BORN file is used, generating " + prefix + ".fc2_lr",flush=True)
    phcel.nac_params = parse_BORN(phcel.primitive, symprec=1e-5, is_symmetry=True,
                                  filename='BORN')
    phcel.force_constants = np.zeros((len(scel), len(scel), 3, 3))
    dynmat = phcel.dynamical_matrix
    dynmat.make_Gonze_nac_dataset()
    fc2_LR = -dynmat.Gonze_nac_dataset[0]

    ## LR FC2 are written in the calculator units!!
    with open(prefix + ".fc2_lr","wb") as f:
        pickle.dump(fc2_LR, f)

# write the FC conversion factor to output
fc_factor = get_force_constant_conversion_factor(units['force_constants_unit'],interface_mode='vasp')
print(f'FC unit conversion factor to eV/ang**2: {fc_factor}',flush=True)

# create the info file
with open(prefix + ".info","wb") as f:
    pickle.dump([calculator.lower(), maximum_cutoff, acoustic_sum_rules, nthread_batch_lsqr, phcalc,
                 ncell, cell, cell_for_cs, scel, fc_factor,phcel,out_kwargs],f)

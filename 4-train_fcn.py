## 4-train_fcn.py: train a force constant model
##
## Input: prefix.info, forces in */, prefix.cs
## -> optional: prefix.fc2_lr ## subtract LR from reference forces first, if file present
## Output: prefix.fcn

## input block ##
prefix="blah" ## prefix for the generated files
outputs="blah*/*.out" # regular expression for the files
fit_method="rfe" # training method
validation_nsplit=5 # number of splits in validation
train_fraction=0.8 # fraction of data used in training/validation split
#################

import os
from glob import glob
import pickle
import numpy as np
import ase
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.utilities import get_displacements
from trainstation import CrossValidationEstimator
from collections import defaultdict
## from phonopy import Phonopy
## from phonopy.structure.atoms import PhonopyAtoms

# load the info file
with open(prefix + ".info","rb") as f:
    ## EB fc_factor added
    calculator, phcalc, ncell, cell, scel, fc_factor, phcel = pickle.load(f)

# read the cluster configuration
with open(prefix + ".cs","rb") as f:
    cutoffs,cs = pickle.load(f)

# read the forces and build the structure container
sc = StructureContainer(cs)
for fname in glob(outputs):
    atoms = ase.io.read(fname)

    # get displacements and forces
    displacements = get_displacements(atoms, scel)
    forces = atoms.get_forces()

    # append to the structure container
    atoms_tmp = scel.copy()
    atoms_tmp.new_array('displacements', displacements)
    atoms_tmp.new_array('forces', forces)
    sc.add_structure(atoms_tmp)

# print out some details
print("--- structure container details ---")
print(sc)
print("")

## remove the long-range contribution from the training data
if os.path.isfile(prefix + ".fc2_lr"):
    print('entro')
    with open(prefix + ".fc2_lr","rb") as f:
        ## fc2_LR written using the calculator units 
        fc2_LR = pickle.load(f) * fc_factor

    displacements = np.array([fs.displacements for fs in sc])
    M, F = sc.get_fit_data()
    F -= np.einsum('ijab,njb->nia', -fc2_LR, displacements).flatten()
else:
    M, F = sc.get_fit_data()
## EB
# fc2_lr ar ok

## run the training
cve = CrossValidationEstimator((M, F),fit_method=fit_method,validation_method='shuffle-split',
                               n_splits=validation_nsplit,train_size=train_fraction,
                               test_size=1-train_fraction)
print("--- training ---")
cve.validate()
print("validation_splits = ",cve.rmse_validation_splits)
cve.train()
print(cve)

## save the force constant potential
fcp = ForceConstantPotential(cs, cve.parameters)
fcp.write(prefix + '.fcn')
print("--- force constant potential details ---")
print(fcp)

## EB already in phcel
## calculate list of temperatures (for checking)
## atoms_phonopy = PhonopyAtoms(symbols=cell.get_chemical_symbols(),
##                              scaled_positions=cell.get_scaled_positions(),
##                              cell=cell.cell)
## ph = Phonopy(atoms_phonopy, supercell_matrix=ncell*np.eye(3),
##              primitive_matrix=None,calculator=phcalc)

fc2 = fcp.get_force_constants(scel).get_fc_array(order=2) 
## EB  TO REMOVE
## if fc2_LR is not None: ##EB is not define if it is none
if os.path.isfile(prefix + ".fc2_lr"):
    fc2 += fc2_LR

## EB return eV/ang**2 to the corresponding units
fc2 = fc2 / fc_factor
## TO REMOVE
from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5
write_FORCE_CONSTANTS(fc2)
## TO REMOVE

phcel.set_force_constants(fc2)
phcel.run_mesh([20] * 3)
phcel.run_thermal_properties(temperatures=300)

fvib = phcel.get_thermal_properties_dict()['free_energy'][0]
svib = phcel.get_thermal_properties_dict()['entropy'][0]
print("\nHarmonic properties at 300 K (kJ/mol): fvib = %.3f svib = %.3f\n" % (fvib,svib))

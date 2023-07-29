## 4-train_fcn.py: train a force constant model
##
## Input: prefix.info, forces in */, prefix.cs
## -> optional: prefix.fc2_lr ## subtract LR from reference forces first, if file present
## Output: prefix.fcn

## input block ##
prefix="blah" ## prefix for the generated files
outputs="blah-*/*out" # regular expression for the files
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
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, phcalc, ncell, cell, scel = pickle.load(f)

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
    with open(prefix + ".fc2_lr","rb") as f:
        fc2_LR = pickle.load(f)
    displacements = np.array([fs.displacements for fs in sc])
    M, F = sc.get_fit_data()
    F -= np.einsum('ijab,njb->nia', -fc2_LR, displacements).flatten()
else:
    M, F = sc.get_fit_data()

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

## calculate list of temperatures (for checking)
atoms_phonopy = PhonopyAtoms(symbols=cell.get_chemical_symbols(),
                             scaled_positions=cell.get_scaled_positions(),
                             cell=cell.cell)
ph = Phonopy(atoms_phonopy, supercell_matrix=ncell*np.eye(3),
             primitive_matrix=None,calculator=phcalc)
fc2 = fcp.get_force_constants(scel).get_fc_array(order=2)
ph.set_force_constants(fc2)
ph.run_mesh([20] * 3)
ph.run_thermal_properties(temperatures=300)
fvib = ph.get_thermal_properties()[1][0]
svib = ph.get_thermal_properties()[2][0]
print("\nAnharmonic properties at 300 K (kJ/mol): fvib = %.3f svib = %.3f\n" % (fvib,svib))

## 4-train_fcn.py: train a force constant model
##
## Input: prefix.info, forces in */, prefix.cs
## -> optional: prefix.fc2_lr ## subtract LR from reference forces first, if file present
## Output: prefix.fcn

## input block ##
prefix="blah" ## prefix for the generated files
outputs=["blah-*/*.out"] # regular expression for the files ()
validation_nsplit=5 # number of splits in validation (set to 0 for plain least-squares)
train_fraction=0.8 # fraction of data used in training/validation split
#################

import os
from glob import glob
import pickle
import numpy as np
import ase
import time
from hiphive import StructureContainer, ForceConstantPotential
from hiphive.utilities import get_displacements
from hiphive_utilities import shuffle_split_cv, least_squares ## M, F , n_splits 10, test_size 0.2

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, phcalc, ncell, cell, cell_for_cs, scel, fc_factor, phcel = pickle.load(f)

# read the cluster configuration
with open(prefix + ".cs","rb") as f:
    cutoffs,cs = pickle.load(f)

# initialize random seed
seed = int(time.time())
print(f'Initialized random seed: {seed}')
rs = np.random.RandomState(seed)

flist = []
if isinstance(outputs,str):
    flist.extend(glob(outputs))
else:
    for i in outputs:
        flist.extend(glob(i))

# read the forces and build the structure container
sc = StructureContainer(cs)
for fname in flist:
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

## remove the long-range contribution from the training data (fc2_LR written using the calculator units)
if os.path.isfile(prefix + ".fc2_lr"):
    with open(prefix + ".fc2_lr","rb") as f:
        fc2_LR = pickle.load(f) * fc_factor

    displacements = np.array([fs.displacements for fs in sc])
    M, F = sc.get_fit_data()
    F -= np.einsum('ijab,njb->nia', -fc2_LR, displacements).flatten()
else:
    M, F = sc.get_fit_data()

## run the training
if (validation_nsplit <= 0):
    opt, coefs, rmse = least_squares(M, F)
else:
    opt, coefs, rmse = shuffle_split_cv(M, F, n_splits=validation_nsplit,
                                      test_size=(1-train_fraction),seed=rs)

# Calculate and print the adjusted r2
r2 = opt.score(M,F)
nparam = opt.n_features_in_
ndata = F.size
ar2 = 1- (1 - r2) * (ndata-1) / (ndata - nparam - 1)
print("Final adjusted R2 = %.8f\n" %(ar2))

## save the force constant potential
fcp = ForceConstantPotential(cs, coefs)
fcp.write(prefix + '.fcn')
print("--- force constant potential details ---")
print(fcp)

## get the fc2, convert from eV/ang**2 to corresponding units
fc2 = fcp.get_force_constants(scel).get_fc_array(order=2)
if os.path.isfile(prefix + ".fc2_lr"):
    fc2 += fc2_LR
fc2 = fc2 / fc_factor

#### write the harmonic fc2 to a file
## from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5
## write_FORCE_CONSTANTS(fc2)

phcel.set_force_constants(fc2)
phcel.run_mesh(150.)
phcel.run_thermal_properties(temperatures=300)

#### write the harmonic phDOS to a file?
## phcel.run_total_dos()
## phcel.write_total_dos(filename='qha-dos.dat')

fvib = phcel.get_thermal_properties_dict()['free_energy'][0]
svib = phcel.get_thermal_properties_dict()['entropy'][0]

print("\nQuality of the fit: RMSE = %.7f meV/ang, avg-abs-F = %.7f meV/ang" % (rmse*1000, np.mean(np.abs(F))*1000))
print("Harmonic properties at 300 K (kJ/mol): fvib = %.3f svib = %.3f\n" % (fvib,svib))

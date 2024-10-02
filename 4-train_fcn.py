## 4-train_fcn.py: train a force constant model
##
## Input: prefix.info, forces in */, prefix.cs
## -> optional: prefix.fc2_lr ## subtract LR from reference forces first, if file present
## Output: prefix.fcn

## input block ##
prefix="mgo" ## prefix for the generated files
outputs=["mgo-*/*.out"] # regular expression for the files ()
#################

import os
from glob import glob
import pickle
import numpy as np
import ase
import time
from hiphive import StructureContainer, ForceConstantPotential
from hiphive.utilities import get_displacements
from hiphive_utilities import shuffle_split_cv, least_squares_batch_simple

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, phcalc, ncell, cell, cell_for_cs, scel, fc_factor, phcel = pickle.load(f)

# read the cluster configuration
with open(prefix + ".cs","rb") as f:
    cutoffs,cs = pickle.load(f)

## read the long-range FC2
fc2_LR = None
if os.path.isfile(prefix + ".fc2_lr"):
    with open(prefix + ".fc2_lr","rb") as f:
        fc2_LR = pickle.load(f) * fc_factor

## run least squares
coefs, rmse, Favgabs, r2, ar2 = least_squares_batch_simple(outputs,cs,scel,fc2_LR)

## save the force constant potential
fcp = ForceConstantPotential(cs, coefs)
fcp.write(prefix + '.fcn')
print("--- force constant potential details ---")
print(fcp)

## get the fc2, convert from eV/ang**2 to corresponding units
fc2 = fcp.get_force_constants(scel).get_fc_array(order=2)
if fc2_LR is not None:
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

print("Mesh shape = ",phcel._mesh._mesh)
print("Negative frequencies in mesh = %d out of %d" % (np.sum(phcel._mesh.frequencies < 0),phcel._mesh.frequencies.size))
print("Quality of the fit: r2 = %.7f, adjusted-r2 = %.7f" % (r2, ar2))
print("Quality of the fit: RMSE = %.7f meV/ang, avg-abs-F = %.7f meV/ang" % (rmse*1000, Favgabs))
print("Harmonic properties at 300 K (kJ/mol): fvib = %.3f svib = %.3f\n" % (fvib,svib))

## 2-calculate_harmonic_fc2.py: train a harmonic (FC2) model to the
## structures generated by simple rattling using least squares. The
## resulting fc2 can be used for the more sophisticated phonon rattle.
##
## Input: prefix.info, forces in */
## -> optional: prefix.fc2_lr ## subtract LR from reference forces before the fit
## Output: prefix.fc2_harmonic

## input block ##
prefix="mgo" ## prefix for the generated files
outputs="harmonic-*/*.out" ## regular expression for the files (QE,aims=*.out,VASP=*.xml)
#################

import pickle
import numpy as np
import os
from hiphive import ClusterSpace, ForceConstantPotential
from hiphive.utilities import get_displacements
from hiphive_utilities import write_negative_frequencies_file, least_squares_batch_simple, has_negative_frequencies

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, use_batch_lsqr, phcalc, ncell, cell, cell_for_cs, scel, fc_factor, phcel = pickle.load(f)

# build cluster space with only fc2
cutoffs = [maximum_cutoff]
cs = ClusterSpace(cell_for_cs, cutoffs, acoustic_sum_rules=acoustic_sum_rules)

## read the long-range fc2, if available
fc2_LR = None
if os.path.isfile(prefix + ".fc2_lr"):
    with open(prefix + ".fc2_lr","rb") as f:
        fc2_LR = pickle.load(f) * fc_factor

## run least squares
coefs, rmse, Favgabs, r2, ar2 = least_squares_batch_simple(outputs,cs,scel,fc2_LR)

## save the force constant potential
fcp = ForceConstantPotential(cs, coefs)
fc2 = fcp.get_force_constants(scel).get_fc_array(order=2)
if fc2_LR is not None:
    fc2 += fc2_LR

# return eV/ag**2 to the corresponding units
fc2 = fc2 / fc_factor

#### various other ways of writing the resulting fc2
## from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5
## write_FORCE_CONSTANTS(fc2)

## fc2 written with the calculator units
with open(prefix + ".fc2_harmonic","wb") as f:
    pickle.dump(fc2, f)

## update phonopy
phcel.force_constants = fc2
phcel.run_mesh(150.,with_eigenvectors=True)
phcel.run_thermal_properties(temperatures=300)
fvib = phcel.get_thermal_properties_dict()['free_energy'][0]
svib = phcel.get_thermal_properties_dict()['entropy'][0]

#print(phcel._mesh.frequencies[phcel._mesh.frequencies < 0])
#print(phcel._mesh.frequencies)
#print(phcel._mesh.qpoints)
#phcel._mesh.write_yaml()
print("Mesh shape = ",phcel._mesh._mesh)
print("Negative frequencies in mesh = %d out of %d" % (np.sum(phcel._mesh.frequencies < 0),phcel._mesh.frequencies.size))
print("Quality of the fit: r2 = %.7f, adjusted-r2 = %.7f" % (r2, ar2))
print("Quality of the fit: RMSE = %.7f meV/ang, avg-abs-F = %.7f meV/ang" % (rmse*1000, Favgabs))
print("Harmonic properties at 300 K: Fvib = %.3f kJ/mol, Svib = %.3f J/K/mol" % (fvib,svib))

## write negative frequencies file
if has_negative_frequencies(phcel._mesh.frequencies):
    filename = prefix + ".fc2_negative_frequencies"
    write_negative_frequencies_file(phcel._mesh,filename)
print()

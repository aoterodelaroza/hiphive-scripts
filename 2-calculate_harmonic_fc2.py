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

from glob import glob
import pickle
import numpy as np
import ase
import os
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.utilities import get_displacements
from hiphive_utilities import least_squares, write_negative_frequencies_file, least_squares_simple

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, phcalc, ncell, cell, scel, fc_factor, phcel = pickle.load(f)

# build cluster space with only fc2
cutoffs = [maximum_cutoff]
cs = ClusterSpace(cell, cutoffs, acoustic_sum_rules=acoustic_sum_rules)

# read the forces and build the structure container
sc = StructureContainer(cs)
for fname in glob(outputs):
    atoms = ase.io.read(fname)

    # this is because otherwise the atoms are not in POSCAR order
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

## run the training
if os.path.isfile(prefix + ".fc2_lr"):
    with open(prefix + ".fc2_lr","rb") as f:
        fc2_LR = pickle.load(f) * fc_factor

    displacements = np.array([fs.displacements for fs in sc])
    M, F = sc.get_fit_data()
    F -= np.einsum('ijab,njb->nia', -fc2_LR, displacements).flatten()
else:
    M, F = sc.get_fit_data()

coefs, rmse = least_squares_simple(M, F)

## save the force constant potential
fcp = ForceConstantPotential(cs, coefs)
fc2 = fcp.get_force_constants(scel).get_fc_array(order=2)
if os.path.isfile(prefix + ".fc2_lr"):
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
phcel.run_mesh(150.)
phcel.run_thermal_properties(temperatures=300)
fvib = phcel.get_thermal_properties_dict()['free_energy'][0]
svib = phcel.get_thermal_properties_dict()['entropy'][0]

#print(phcel._mesh.frequencies[phcel._mesh.frequencies < 0])
#print(phcel._mesh.frequencies)
#print(phcel._mesh.qpoints)
print("Mesh shape = ",phcel._mesh._mesh)
print("Negative frequencies in mesh = %d out of %d" % (np.sum(phcel._mesh.frequencies < 0),phcel._mesh.frequencies.size))
print("Quality of the fit: RMSE = %.7f meV/ang, avg-abs-F = %.7f meV/ang" % (rmse*1000, np.mean(np.abs(F))*1000))
print("Harmonic properties at 300 K: Fvib = %.3f kJ/mol, Svib = %.3f J/K/mol" % (fvib,svib))

## write negative frequencies file
if np.sum(phcel._mesh.frequencies < 0) > 0:
    filename = prefix + ".fc2_negative_frequencies"
    write_negative_frequencies_file(phcel._mesh,filename)
print()

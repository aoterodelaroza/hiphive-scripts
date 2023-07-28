## 3-calculate_effective_frequencies.py: calculate effective frequencies from FCn
##
## Input: prefix.info, prefix.cs, prefix.fcn
## -> optional: prefix.fc2_lr ## subtract LR from reference forces first, if file present
## Output: prefix.svib

## input block ##
prefix="blah" ## prefix for the generated files
cutoff_2nd=None ## cutoff for the structure generation; None=second-order cutoff in clusterspace
n_structures = 30 # number of structures used in scph
train_fraction=0.8 # fraction of data used in training/validation split
n_iterations = 10 # iterations in scph, at least 5
fit_method="rfe" # training method
temperatures=list(range(250,3001,250)) # temperature list (0 is always included)
#################

import os
import time
import pickle
import numpy as np
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.calculators import ForceConstantCalculator
from hiphive.force_constant_model import ForceConstantModel
from hiphive.structure_generation import generate_rattled_structures, generate_phonon_rattled_structures
from hiphive.utilities import prepare_structures
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from trainstation import Optimizer

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, ncell, cell, scel = pickle.load(f)

# load the cs file
with open(prefix + ".cs","rb") as f:
    cutoffs,cs = pickle.load(f)

# load the fcp
fcp = ForceConstantPotential.read(prefix + '.fcn')

# load the LR fc2, if present
fc2_LR = None
if os.path.isfile(prefix + ".fc2_lr"):
    with open(prefix + ".fc2_lr","rb") as f:
        fc2_LR = pickle.load(f)

# build the cs for the schp calculation, force constants, and calculator
if not cutoff_2nd:
    cutoff_2nd = cutoffs[0]
cs = ClusterSpace(cell,[cutoff_2nd])
fcs = fcp.get_force_constants(scel)
calc = ForceConstantCalculator(fcs)

# prepare the phonopy object for the fvib/svib calculation
atoms_phonopy = PhonopyAtoms(symbols=cell.get_chemical_symbols(),
                             scaled_positions=cell.get_scaled_positions(),
                             cell=cell.cell)
ph = Phonopy(atoms_phonopy, supercell_matrix=ncell*np.eye(3),
             primitive_matrix=None,calculator=calculator)

# open output file
fout = open(prefix + ".svib","w")

# calculate the harmonic quantities
fc2 = fcp.get_force_constants(scel).get_fc_array(order=2)
if fc2_LR is not None:
    fc2 += fc2_LR
ph.set_force_constants(fc2)
ph.run_mesh([20] * 3)
ph.run_thermal_properties(temperatures=0)
fvib = ph.get_thermal_properties()[1][0]
svib = ph.get_thermal_properties()[2][0]
print("\nHarmonic properties at zero temperature (kJ/mol): fvib = %.3f svib = %.3f\n" % (fvib,svib))
print("# Anharmonic thermodynamic properties calculated with scph",file=fout)
print("# F does not contain the anharmonic term",file=fout)
print("# T in K, F in kJ/mol, S in J/K/mol",file=fout)
print("# T Fvib Fvibstd Svib Svibstd",file=fout)
print("%.2f %.8f %.8f %.8f %.8f" % (0,fvib,0,svib,0),file=fout)

# initialize structure container and force constant model
sc = StructureContainer(cs)
fcm = ForceConstantModel(scel, cs)

# generate initial model
rattled_structures = generate_rattled_structures(scel, n_structures, 0.03)
rattled_structures = prepare_structures(rattled_structures, scel, calc)
for structure in rattled_structures:
    sc.add_structure(structure)
seed = int(time.time())
print("Random seed = %d" % seed)
opt = Optimizer(sc.get_fit_data(),fit_method=fit_method,train_size=train_fraction,seed=seed)
opt.train()
sc.delete_all_structures()

# run poor man's self consistent
for t in temperatures:
    print("\nStarted scph at temperature: %.2f K" % t);
    param_old = opt.parameters.copy()
    param_last = None
    flist = []
    slist = []
    for i in range(max(n_iterations,5)):
        # generate structures with new FC2, including the LR correction
        fcm.parameters = param_old
        fc2 = fcm.get_force_constants().get_fc_array(order=2, format='ase')
        if fc2_LR is not None:
            fc2 += fc2_LR
        phonon_rattled_structures = generate_phonon_rattled_structures(scel,fc2,n_structures,t)

        # calculate forces with FCn, without LR correction
        phonon_rattled_structures = prepare_structures(phonon_rattled_structures, scel, calc)

        # build the new structurecontainer and fit new model
        for structure in phonon_rattled_structures:
            sc.add_structure(structure)
        opt = Optimizer(sc.get_fit_data(),fit_method=fit_method,train_size=train_fraction,seed=seed+i)
        opt.train()
        sc.delete_all_structures()

        # update parameters
        alpha = 0.2
        param_new = alpha * opt.parameters + (1-alpha) * param_old

        # calculate fvib
        fc2 = ForceConstantPotential(cs,param_new).get_force_constants(scel).get_fc_array(order=2)
        if fc2_LR is not None:
            fc2 += fc2_LR
        ph.set_force_constants(fc2)
        ph.run_mesh([20] * 3)
        ph.run_thermal_properties(temperatures=[t])
        fvib = ph.get_thermal_properties()[1][0]
        svib = ph.get_thermal_properties()[2][0]

        # print iteration summary
        disps = [atoms.get_array('displacements') for atoms in phonon_rattled_structures]
        disp_ave = np.mean(np.abs(disps))
        disp_max = np.max(np.abs(disps))
        x_new_norm = np.linalg.norm(param_new)
        delta_x_norm = np.linalg.norm(param_old-param_new)
        print('{}: x_new = {:.3e}, delta_x = {:.3e}, disp_ave = {:.5f} fvib = {:.3f} svib = {:.3f}'.format(i,x_new_norm, delta_x_norm, disp_ave, fvib, svib))
        param_last = param_new
        param_old = param_new
        flist.append(fvib)
        slist.append(svib)

    # standard deviation of fvib and svib and final message
    fvibstd = np.std(flist)
    svibstd = np.std(slist)
    print("Converged (K,kJ/mol,J/K/mol): T = %.2f fvib = %.3f fvibstd = %.5f svib = %.3f svibstd = %.5f" % (t,fvib,fvibstd,svib,svibstd))
    print("%.2f %.8f %.8f %.8f %.8f" % (t,fvib,fvibstd,svib,svibstd),file=fout)

# clean up
fout.close()

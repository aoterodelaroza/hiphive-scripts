## 5-calculate_effective_frequencies.py: calculate effective frequencies from FCn
##
## Input: prefix.info, prefix.cs, prefix.fcn
## -> optional: prefix.fc2_lr ## subtract LR from reference forces first, if file present
## Output: prefix.svib
## -> optional: prefix-<temperature>.fc2_eff ## per-temperature second-order effective force constants
##
## Stop if imaginary frequencies appear.

import numpy as np

## input block ##
prefix="blah" ## prefix for the generated files
n_structures = 10 # number of structures used in scph
train_fraction=0.8 # fraction of data used in training/validation split
fit_method="rfe" # training method
temperatures = [300, 3000] # temperature list (0 is always included) eg: np.arange(440, 0, -10)
write_fc2eff = True # write the second-order effective force constants file (prefix-temp.fc2_eff)
restart_fc2 = None # name of the FC2 file to start from or None
alpha = 0.2 # damping factor for the parameters in the scph iterations
n_max = 300 # max number of steps in scph
n_safe = 30 # minimum number of steps required to have real frequencies before averaging
n_dead = 20 # crash after n_dead with imaginary frequencies
n_last = 20 # n_last steps are used for fvib, svib, etc. averages
#################

import os
import time
import pickle
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.cutoffs import estimate_maximum_cutoff
from hiphive.calculators import ForceConstantCalculator
from hiphive.force_constant_model import ForceConstantModel
from hiphive.structure_generation import generate_rattled_structures, generate_phonon_rattled_structures
from hiphive.utilities import prepare_structures
from trainstation import Optimizer
from hiphive import ForceConstants

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, phcalc, ncell, cell, scel, fc_factor, phcel = pickle.load(f)

# load the cs file
## EB se carga pero no se usa
with open(prefix + ".cs","rb") as f:
    cutoffs,cs = pickle.load(f)

# load the fcp
fcp = ForceConstantPotential.read(prefix + '.fcn')

# load the LR fc2, if present
fc2_LR = None
if os.path.isfile(prefix + ".fc2_lr"):
    with open(prefix + ".fc2_lr","rb") as f:
        fc2_LR = pickle.load(f)
        fc2_LR = fc2_LR * fc_factor ## return eV/Ang**2
# build the cs for the schp calculation, force constants, and calculator
cs = ClusterSpace(cell,[estimate_maximum_cutoff(scel)-1e-4])
fcs = fcp.get_force_constants(scel)
calc = ForceConstantCalculator(fcs)

# just in case an FC file is read automatically by phonopy
phcel.force_constants = np.zeros((len(scel), len(scel), 3, 3))

# open output file
fout = open(f'{prefix}.svib' ,"w")
if restart_fc2 is not None:
    fc2 = ForceConstants.read_phonopy(supercell=scel,fname=retart_fc2,format='hdf5')
    fc2 = fc2.get_fc_array(order=2) * fc_factor
else:
    fc2 = fcp.get_force_constants(scel).get_fc_array(order=2)

if fc2_LR is not None:
    fc2 += fc2_LR

# calculate the harmonic quantities
phcel.force_constants = fc2 / fc_factor  ## fc2 still in eV/ang**2
phcel.run_mesh([20] * 3)
phcel.run_thermal_properties(temperatures=0)
fvib = phcel.get_thermal_properties_dict()['free_energy'][0]
svib = phcel.get_thermal_properties_dict()['entropy'][0]

print("\nHarmonic properties at zero temperature (kJ/mol): fvib = %.3f svib = %.3f\n" % (fvib,svib))
print("# Anharmonic thermodynamic properties calculated with scph using",
      f"{n_structures} structures and {n_safe} iterations",file=fout)
print("# F does not contain the anharmonic term",file=fout)
print("# T in K, F in kJ/mol, S in J/K/mol",file=fout)
print("# T Fvib Fvibstd Svib Svibstd",file=fout)
print("%.2f %.8f %.8f %.8f %.8f" % (0,fvib,0,svib,0),file=fout)

# initialize structure container and force constant model
sc = StructureContainer(cs)
fcm = ForceConstantModel(scel, cs)

# generate initial model
rattled_structures = generate_rattled_structures(scel, n_structures, 0.15)
rattled_structures = prepare_structures(rattled_structures, scel, calc, check_permutation=False)
for structure in rattled_structures:
    sc.add_structure(structure)
seed = int(time.time())
print("# Random seed = %d" % seed)

opt = Optimizer(sc.get_fit_data(),fit_method=fit_method,train_size=train_fraction,seed=seed)
opt.train()
sc.delete_all_structures()

# run poor man's self consistent phonon frequencies
for t in temperatures:
    print("\nStarted scph at temperature: %.2f K" % t);
    param_old = opt.parameters.copy()
    flist = []
    slist = []
    live_counter = 0
    dead_counter = 0
    for i in range(max(n_max,1)):

        # generate structures with new FC2, including the LR correction
        fcm.parameters = param_old
        fc2 = fcm.get_force_constants().get_fc_array(order=2)
        if fc2_LR is not None:
            fc2 += fc2_LR
        phonon_rattled_structures = generate_phonon_rattled_structures(scel,fc2,n_structures,t)

        # calculate forces with FCn, without LR correction
        phonon_rattled_structures = prepare_structures(phonon_rattled_structures, scel, calc, check_permutation=False)

        # build the new structurecontainer and fit new model
        for structure in phonon_rattled_structures:
            sc.add_structure(structure)
        opt = Optimizer(sc.get_fit_data(), fit_method=fit_method,
                train_size=train_fraction, seed=seed+i)

        opt.train()
        sc.delete_all_structures()

        # calculate fvib
        param_new = alpha * opt.parameters + (1-alpha) * param_old
        fc2 = ForceConstantPotential(cs, param_new).get_force_constants(scel).get_fc_array(order=2) # only short-range

        if fc2_LR is not None:
            fc2 += fc2_LR
        phcel.force_constants = fc2 / fc_factor  ## fc2 still in eV/ang**2
        phcel.run_mesh([20] * 3)
        phcel.run_thermal_properties(temperatures=[t])

        x_new_norm = np.linalg.norm(param_new)
        delta_x_norm = np.linalg.norm(param_old-param_new)

        ## check whether to stop if negative/imaginary frequencies
        fvib = phcel.get_thermal_properties_dict()['free_energy'][0]
        if np.isnan(fvib):
            if dead_counter == n_dead//2:
                live_counter = 0
            rand = 0.4
            param_old = rand * opt.parameters + (1-rand)*param_old
            dead_counter += 1
        else:
            if live_counter == n_dead // 2:
                dead_counter = 0
            svib = phcel.get_thermal_properties_dict()['entropy'][0]
            flist.append(fvib)
            slist.append(svib)
            param_old = param_new
            live_counter += 1

        ## print iteration summary
        disps = [atoms.get_array('displacements') for atoms in phonon_rattled_structures]
        disp_ave = np.mean(np.abs(disps))
        disp_max = np.max(np.abs(disps))
        print(f'{i}: x_new = {x_new_norm:.3e}, delta_x = {delta_x_norm:.3e},',
              f'disp_ave = {disp_ave:.5f}, fvib = {fvib:.3f},',
              f'svib = {svib:.3f}')

        if dead_counter == n_dead:
            print(f'Frequencies not converged at temperature {t}, loop ended')
            exit()

        if i == n_max-1:
            print(f'Frequencies not converged at tempearture {t}, loop ended')
            exit()

        if live_counter == n_safe:
            ## print last values only
            fvib = np.mean(flist[len(flist)-n_last:len(flist)])
            svib = np.mean(slist[len(slist)-n_last:len(slist)])
            fvibstd = np.std(flist[len(flist)-n_last:len(flist)])
            svibstd = np.std(slist[len(slist)-n_last:len(slist)])
            break

    # standard deviation of fvib and svib and final message
    print("Converged (K,kJ/mol,J/K/mol): T = %.2f fvib = %.3f fvibstd = %.5f svib = %.3f svibstd = %.5f" % (t,fvib,fvibstd,svib,svibstd))
    print("%.2f %.8f %.8f %.8f %.8f" % (t,fvib,fvibstd,svib,svibstd),file=fout)
    if write_fc2eff == True:
        fc2 = ForceConstants.from_arrays(scel, fc2_array=(fc2 / fc_factor), fc3_array=None)
        fc2.write_to_phonopy(f'./{prefix}-{t:04d}.fc2_eff', format='hdf5')
fout.close()

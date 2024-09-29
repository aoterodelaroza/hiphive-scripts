## 5-calculate_effective_frequencies.py: calculate effective frequencies on a temperature grid from FCn
##
## Input: prefix.info, prefix.fcn
## -> optional: prefix.fc2_lr ## subtract LR from reference forces first, if file present
## Output: prefix.svib
## -> optional: prefix-<temperature>.fc2_eff ## per-temperature second-order effective force constants
##
## Stop if imaginary frequencies appear.

import numpy as np

## input block ##
prefix="blah" ## prefix for the generated files
n_structures = 10 # number of structures used in scph
validation_nsplit=0 # number of splits in validation (set to 0 for plain least-squares)
train_fraction=0.8 # fraction of data used in training/validation split
temperatures = np.arange(100, 2700, 100) # temperature list (0 is always included) eg: np.arange(440, 0, -10)
write_fc2eff = False # write the second-order effective force constants file (prefix-temp.fc2_eff)
#################

## details of SCPH ##
alpha = 0.2 # damping factor for the parameters in the scph iterations
n_max = 300 # max number of steps in scph
n_safe = 50 # minimum number of steps required to have real frequencies before averaging
n_dead = 35 # crash after n_dead with imaginary frequencies
n_last = 40 # n_last steps are used for fvib, svib, etc. averages
#################

import os
import time
import pickle
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive import ForceConstants
from hiphive.calculators import ForceConstantCalculator
from hiphive.force_constant_model import ForceConstantModel
from hiphive.utilities import prepare_structures
from hiphive.structure_generation import  generate_phonon_rattled_structures
from hiphive_utilities import constant_rattle, shuffle_split_cv, least_squares

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, phcalc, ncell, cell, scel, fc_factor, phcel = pickle.load(f)

# initialize random seed
seed = int(time.time())
print(f'Initialized random seed: {seed}')
rs = np.random.RandomState(seed)

# read fcn and fc2_lr
fcp = ForceConstantPotential.read(f'{prefix}.fcn')
fc2_LR = None
if os.path.isfile(prefix + ".fc2_lr"):
    with open(prefix + ".fc2_lr","rb") as f:
        fc2_LR = pickle.load(f) * fc_factor

# just in case an FC file is read automatically by phonopy
cs = ClusterSpace(cell,[maximum_cutoff], acoustic_sum_rules=acoustic_sum_rules)
fcs = fcp.get_force_constants(scel)
calc = ForceConstantCalculator(fcs)
phcel.force_constants = np.zeros((len(scel), len(scel), 3, 3))
# To avoid destroy previous results
if os.path.isfile(f'{prefix}.svib'):
    date = time.strftime("%d-%b-%Y-%H:%M:%S", time.gmtime())
    os.rename(f'{prefix}.svib', f'{prefix}-{date}.svib')

fout = open(f'{prefix}.svib' ,"w")

fc2 = fcp.get_force_constants(scel).get_fc_array(order=2)
if fc2_LR is not None:
    fc2 += fc2_LR

# calculate the harmonic quantities
phcel.force_constants = fc2 / fc_factor  ## fc2 still in eV/ang**2
phcel.run_mesh(150.)
phcel.run_thermal_properties(temperatures=0)
fvib = phcel.get_thermal_properties_dict()['free_energy'][0]
svib = phcel.get_thermal_properties_dict()['entropy'][0]

print("\nHarmonic properties at zero temperature (kJ/mol): fvib = %.3f svib = %.3f\n" % (fvib,svib))
##EB print iterations and structures
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
rattled_structures = constant_rattle(scel, n_structures, 0.15, rs)
rattled_structures = prepare_structures(rattled_structures, scel, calc, check_permutation=False)

for structure in rattled_structures:
    sc.add_structure(structure)
M , F = sc.get_fit_data()
_, coefs, rmse = least_squares(M, F, verbose=0)

sc.delete_all_structures()

# run poor man's self consistent phonon frequencies
for t in temperatures:
    print("\nStarted scph at temperature: %.2f K" % t);
    param_old = coefs.copy()
    flist, slist = [], []
    counter_n_max, counter_n_dead, counter_n_safe = 0, 0, 0 
    while counter_n_max < n_max:
        counter_n_max += 1
        # generate structures with new FC2, including the LR correction
        fcm.parameters = param_old
        fc2 = fcm.get_force_constants().get_fc_array(order=2)

        if os.path.isfile(prefix + ".fc2_lr"):
            fc2 += fc2_LR
        phonon_rattled_structures = generate_phonon_rattled_structures(scel,fc2,n_structures,t)
        # calculate forces with FCn, without LR correction
        phonon_rattled_structures = prepare_structures(phonon_rattled_structures, scel, calc, check_permutation=False)
 
        # build the new structurecontainer and fit new model
        for structure in phonon_rattled_structures:
            sc.add_structure(structure)
 
        M , F = sc.get_fit_data()
        if (validation_nsplit == 0):
            _, coefs, rmse = least_squares(M, F, verbose=0)
        else:
            _, coefs, rmse = shuffle_split_cv(M, F, n_splits=validation_nsplit,
                                              test_size=(1 -train_fraction),seed=rs,verbose=0)
        sc.delete_all_structures()
 
        # calculate fvib
        param_new = alpha * coefs + (1-alpha) * param_old
        fc2 = ForceConstantPotential(cs, param_new).get_force_constants(scel).get_fc_array(order=2) # only short-range
        if os.path.isfile(prefix + ".fc2_lr"):
            fc2 += fc2_LR
 
        phcel.force_constants = fc2 / fc_factor  ## fc2 still in eV/ang**2
        phcel.run_mesh(150.)
        phcel.run_thermal_properties(temperatures=[t], cutoff_frequency=-10.0)
        x_new_norm = np.linalg.norm(param_new)
        delta_x_norm = np.linalg.norm(param_old-param_new)
        ## check whether to stop if negative/imaginary frequencies
        fvib = phcel.get_thermal_properties_dict()['free_energy'][0]
        svib = phcel.get_thermal_properties_dict()['entropy'][0]
        ## counter_n_max, counter_n_dead, counter_n_safe
        # if svib is not real
        if np.isnan(svib):
            counter_n_dead += 1
            ## harder criteria for safe results
            if counter_n_dead >= n_safe*0.75:
                counter_n_safe = 0
            param_old = 0.4 * coefs + (1 - 0.4)*param_old
        else:
            counter_n_safe += 1
            ## store Fvib and Svib
            flist.append(fvib)
            slist.append(svib)
            ## update the model
            param_old = param_new
        ## print iteration summary
        disps = [atoms.get_array('displacements') for atoms in phonon_rattled_structures]
        disp_ave = np.mean(np.abs(disps))
        disp_max = np.max(np.abs(disps))
        print(f'{counter_n_max}: x_new = {x_new_norm:.3e},',
              f'delta_x = {delta_x_norm:.3e},',
              f'disp_ave = {disp_ave:.5f}, fvib = {fvib:.3f},',
              f'svib = {svib:.3f}')
        if counter_n_dead == n_dead:
            print(f'Frequencies not converged at temperature {t}')
            break
        if counter_n_safe == n_safe:
        #     ## print last values only
            fvib = np.mean(flist[len(flist)-n_last:len(flist)])
            svib = np.mean(slist[len(slist)-n_last:len(slist)])
            fvibstd = np.std(flist[len(flist)-n_last:len(flist)])
            svibstd = np.std(slist[len(slist)-n_last:len(slist)])
            print("Converged (K,kJ/mol,J/K/mol): T = %.2f fvib = %.3f fvibstd = %.5f svib = %.3f svibstd = %.5f" % (t,fvib,fvibstd,svib,svibstd))
            ## write the results
            print("%.2f %.8f %.8f %.8f %.8f" % (t,fvib,fvibstd,svib,svibstd),file=fout)
            break
    if write_fc2eff == True:
        fc2 = ForceConstants.from_arrays(scel, fc2_array=(fc2 / fc_factor),
                                         fc3_array=None)
        fc2.write_to_phonopy(f'./{prefix}-{t:04d}.fc2_eff', format='hdf5')

    if counter_n_max == n_max:
        print(f'Frequencies not converged at temperature {t} max iterations reached')
fout.close()

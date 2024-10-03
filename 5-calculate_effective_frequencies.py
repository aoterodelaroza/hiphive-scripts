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
prefix="mgo" ## prefix for the generated files
n_structures = 10 # number of structures used in scph
temperatures = np.arange(100, 2700, 100) # temperature list (0 is always included) eg: np.arange(440, 0, -10)
write_fc2eff = False # write the second-order effective force constants file (prefix-temp.fc2_eff)
#################

## details of SCPH ##
alpha = 0.1 # damping factor for the parameters in the scph iterations
n_max = 35 # max number of steps in scph
n_last = 10 # n_last steps are used for fvib, svib, etc. averages
#################

import os
import sys
import time
import pickle
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive import ForceConstants
from hiphive.calculators import ForceConstantCalculator
from hiphive.force_constant_model import ForceConstantModel
from hiphive.utilities import prepare_structures
from hiphive_utilities import constant_rattle, shuffle_split_cv,\
    write_negative_frequencies_file, generate_phonon_rattled_structures, has_negative_frequencies,\
    least_squares_batch_simple

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, use_batch_lsqr, phcalc, ncell, cell, cell_for_cs, scel, fc_factor, phcel = pickle.load(f)

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
cs = ClusterSpace(cell_for_cs,[maximum_cutoff], acoustic_sum_rules=acoustic_sum_rules)
fcs = fcp.get_force_constants(scel)
calc = ForceConstantCalculator(fcs)
phcel.force_constants = np.zeros((len(scel), len(scel), 3, 3))

# open the output svib file
fout = open(f'{prefix}.svib' ,"w")

# initialize the fc2
fc2 = fcp.get_force_constants(scel).get_fc_array(order=2)
if fc2_LR is not None:
    fc2 += fc2_LR

# calculate the harmonic properties from the initial fc2
phcel.force_constants = fc2 / fc_factor  ## fc2 still in eV/ang**2
phcel.run_mesh(150.)
phcel.run_thermal_properties(temperatures=0)
fvib = phcel.get_thermal_properties_dict()['free_energy'][0]
svib = phcel.get_thermal_properties_dict()['entropy'][0]
print("\nHarmonic properties at zero temperature (kJ/mol): fvib = %.3f svib = %.3f\n" % (fvib,svib))

# header
print("# Anharmonic thermodynamic properties calculated with scph using",
      f"{n_structures} structures and {n_max} iterations",file=fout)
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

# calculate the first least squares for the initial parameters
for s in rattled_structures:
    sc.add_structure(s)
coefs, _, _, _, _ = least_squares_batch_simple(sc,cs,scel,skiprmse=1)
sc.delete_all_structures()

# run poor man's self consistent phonon frequencies
for t in temperatures:
    print("\nStarted scph at temperature: %.2f K" % t);
    param_old = coefs.copy()
    flist, slist = [], []

    # run the number of steps indicated by user
    for it in range(n_max):
        # generate structures with new FC2, including the LR correction
        fcm.parameters = param_old
        fc2 = fcm.get_force_constants().get_fc_array(order=2)
        if os.path.isfile(prefix + ".fc2_lr"):
            fc2 += fc2_LR

        # generate phonon rattled structures with the current fc2
        rattled_structures = generate_phonon_rattled_structures(scel,fc2,n_structures,t)

        # calculate forces at the generated structures
        rattled_structures = prepare_structures(rattled_structures, scel, calc)

        for s in rattled_structures:
            sc.add_structure(s)
        coefs, _, _, _, _ = least_squares_batch_simple(sc,cs,scel,skiprmse=1)
        sc.delete_all_structures()

        # mix the new FC2 with the previous one
        param_new = alpha * coefs + (1-alpha) * param_old
        fc2 = ForceConstantPotential(cs, param_new).get_force_constants(scel).get_fc_array(order=2) # only short-range
        if os.path.isfile(prefix + ".fc2_lr"):
            fc2 += fc2_LR

        # calculate thermodynamic properties
        phcel.force_constants = fc2 / fc_factor  ## fc2 still in eV/ang**2
        phcel.run_mesh(150.)
        phcel.run_thermal_properties(temperatures=[t])

        # check if negative frequencies are present; if so, write negative frequencies file
        has_neg = has_negative_frequencies(phcel._mesh.frequencies)
        if has_neg:
            tint = int(round(t))
            filename = f'{prefix}-{tint:04d}.fc2_negative_frequencies'
            write_negative_frequencies_file(phcel._mesh,filename)
            negstr = "(NEG)"
        else:
            negstr = "     "

        # calculate convergence parameters, calculate and store fvib, svib
        x_new_norm = np.linalg.norm(param_new)
        delta_x_norm = np.linalg.norm(param_old-param_new)
        fvib = phcel.get_thermal_properties_dict()['free_energy'][0]
        svib = phcel.get_thermal_properties_dict()['entropy'][0]
        flist.append(fvib)
        slist.append(svib)

        # update the model
        param_old = param_new

        # print iteration summary
        disps = [atoms.get_array('displacements') for atoms in rattled_structures]
        disp_ave = np.mean(np.abs(disps))
        disp_max = np.max(np.abs(disps))
        print(f'{it}: {negstr} x_new = {x_new_norm:.3e},',
              f'delta_x = {delta_x_norm:.3e},',
              f'disp_ave = {disp_ave:.5f}, fvib = {fvib:.3f},',
              f'svib = {svib:.3f}')
        sys.stdout.flush()

    # calculate average properties and output
    fvib = np.mean(flist[len(flist)-n_last:len(flist)])
    svib = np.mean(slist[len(slist)-n_last:len(slist)])
    fvibstd = np.std(flist[len(flist)-n_last:len(flist)])
    svibstd = np.std(slist[len(slist)-n_last:len(slist)])
    if has_neg:
        print("Converged [NEGATIVE] (K,kJ/mol,J/K/mol): T = %.2f fvib = %.3f fvibstd = %.5f svib = %.3f svibstd = %.5f" % (t,fvib,fvibstd,svib,svibstd))
        print("%.2f %.8f %.8f %.8f %.8f (NEG)" % (t,fvib,fvibstd,svib,svibstd),file=fout)
    else:
        print("Converged (K,kJ/mol,J/K/mol): T = %.2f fvib = %.3f fvibstd = %.5f svib = %.3f svibstd = %.5f" % (t,fvib,fvibstd,svib,svibstd))
        print("%.2f %.8f %.8f %.8f %.8f" % (t,fvib,fvibstd,svib,svibstd),file=fout)

    if write_fc2eff == True:
        fc2 = ForceConstants.from_arrays(scel, fc2_array=(fc2 / fc_factor),
                                         fc3_array=None)
        fc2.write_to_phonopy(f'./{prefix}-{t:04d}.fc2_eff', format='hdf5')
fout.close()

## 5-calculate_effective_frequencies.py: calculate effective frequencies on a temperature grid from FCn
##
## Input: prefix.info, prefix.fcn, prefix.cs_harmonic
## -> optional: prefix.fc2_lr ## subtract LR from reference forces first, if file present
## Output: prefix.svib
## -> optional: prefix-<temperature>.fc2_eff ## per-temperature second-order effective force constants
##
## Stop if imaginary frequencies appear.

import numpy as np

## input block ##
prefix="urea" ## prefix for the generated files
temperatures = np.arange(10,500,50) # temperature list (0 is always included) eg: np.arange(440, 0, -10)
write_fc2eff = False # write the second-order effective force constants file (prefix-temp.fc2_eff)
#################

## details of SCPH ##
alpha = [0.1,0.01] # damping factors for the parameters in the scph iterations (fast,slow)
conv_thr = [1e-3,1e-5] # If np.abs(np.sum(np.diff(s[-n_last:]))) / np.mean(s[-n_last:]) < conv_thr, switch alpha (0) or stop the iterations (1)
n_max = 500 # max number of steps in scph
n_last = 10 # n_last steps are used for fvib, svib, etc. averages
#################

import os
import time
import pickle
from hiphive import ForceConstantPotential, ForceConstants
from hiphive.calculators import ForceConstantCalculator
from hiphive.force_constant_model import ForceConstantModel
from hiphive_utilities import constant_rattle,\
    write_negative_frequencies_file, generate_phonon_rattled_structures, has_negative_frequencies,\
    least_squares_batch, least_squares_accum

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, nthread_batch_lsqr, phcalc, ncell, cell, cell_for_cs, scel, fc_factor, phcel, out_kwargs = pickle.load(f)

# initialize random seed
seed = int(time.time())
print(f'Initialized random seed: {seed}',flush=True)
rs = np.random.RandomState(seed)

# read fcn and fc2_lr
fcp = ForceConstantPotential.read(f'{prefix}.fcn')
fc2_LR = None
if os.path.isfile(prefix + ".fc2_lr"):
    with open(prefix + ".fc2_lr","rb") as f:
        fc2_LR = pickle.load(f) * fc_factor

# read the harmonic cluster space, calculate the number of structures
with open(prefix + ".cs_harmonic","rb") as f:
    cutoffs_harmonic,cs_harmonic = pickle.load(f)
n_structures = np.ceil(cs_harmonic.n_dofs * 10.0 / (3 * len(scel))).astype(int)
print("Number of structures: %d" % n_structures,flush=True)

# prepare the force constant calculator, clean phcel force constants
# in case an FC file is read automatically by phonopy
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
print("\nHarmonic properties at zero temperature (kJ/mol): fvib = %.3f svib = %.3f\n" % (fvib,svib),flush=True)

# header
print("# Anharmonic thermodynamic properties calculated with scph using",
      f"{n_structures} structures and {n_max} iterations",file=fout,flush=True)
print("# F does not contain the anharmonic term",file=fout,flush=True)
print("# T in K, F in kJ/mol, S in J/K/mol",file=fout,flush=True)
print("# T Fvib Fvibstd Svib Svibstd",file=fout,flush=True)
print("%.2f %.8f %.8f %.8f %.8f" % (0,fvib,0,svib,0),file=fout,flush=True)

# initialize force constant model
fcm = ForceConstantModel(scel, cs_harmonic)

# generate initial set of structures (0.15 ang in amplitude), add the calculator to each structure
rattled_structures = constant_rattle(scel, n_structures, 0.15, rs)
for istr in rattled_structures:
    istr.calc = calc

# calculate the first least squares for the initial parameters
if nthread_batch_lsqr and nthread_batch_lsqr > 0:
    coefs, _, _, _, _ = least_squares_batch(rattled_structures,nthread_batch_lsqr,cs_harmonic,scel,skiprmse=1)
else:
    coefs, _, _, _, _ = least_squares_accum(rattled_structures,cs_harmonic,scel,skiprmse=1)

# run poor man's self consistent phonon frequencies
alpha0 = alpha[0]
conv_thr0 = conv_thr[0]
for t in temperatures:
    print("\nStarted scph at temperature: %.2f K" % t,flush=True)
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
        for istr in rattled_structures:
            istr.calc = calc

        # least squares
        if nthread_batch_lsqr and nthread_batch_lsqr > 0:
            coefs, _, _, _, _ = least_squares_batch(rattled_structures,nthread_batch_lsqr,cs_harmonic,scel,skiprmse=1)
        else:
            coefs, _, _, _, _ = least_squares_accum(rattled_structures,cs_harmonic,scel,skiprmse=1)

        # mix the new FC2 with the previous one
        param_new = alpha0 * coefs + (1-alpha0) * param_old
        fc2 = ForceConstantPotential(cs_harmonic, param_new).get_force_constants(scel).get_fc_array(order=2) # only short-range
        if os.path.isfile(prefix + ".fc2_lr"):
            fc2 += fc2_LR

        # calculate thermodynamic properties
        phcel.force_constants = fc2 / fc_factor  ## fc2 still in eV/ang**2
        phcel.run_mesh(150.)
        phcel.run_thermal_properties(temperatures=[t])

        # check if negative frequencies are present; if so, write negative frequencies file
        has_neg = has_negative_frequencies(phcel._mesh.frequencies)
        if has_neg:
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

        if has_neg:
            tint = int(round(t))
            filename = f'{prefix}-{tint:04d}.fc2_negative_frequencies'
            write_negative_frequencies_file(phcel._mesh,filename)

        # update the model
        param_old = param_new

        # print iteration summary
        disps = [atoms.get_array('displacements') for atoms in rattled_structures]
        disp_ave = np.mean(np.abs(disps))
        disp_max = np.max(np.abs(disps))
        print(f'{it}: {negstr} x_new = {x_new_norm:.3e},',
              f'delta_x = {delta_x_norm:.3e},',
              f'disp_ave = {disp_ave:.5f}, fvib = {fvib:.3f},',
              f'svib = {svib:.3f}',flush=True)

        if (len(slist) > n_last-1):
            xconv = np.abs(np.sum(np.diff(slist[-n_last:]))) / np.mean(slist[-n_last:])
            if (xconv < conv_thr[1]):
                print("CONVERGED abs(sum(diff(s[-n_last:]))) / mean(s[-n_last:]) = %.5f < conv_thr = %.5f" % (xconv,conv_thr0),flush=True)
                break
            elif (xconv < conv_thr[0]):
                alpha0 = alpha[1]
                conv_thr0 = conv_thr[1]
            print("alpha=%.2e , abs(sum(diff(s[-n_last:]))) / mean(s[-n_last:]) = %.5f < conv_thr = %.5f" % (alpha0,xconv,conv_thr0),flush=True)

    # calculate average properties and output
    fvib = np.mean(flist[len(flist)-n_last:len(flist)])
    svib = np.mean(slist[len(slist)-n_last:len(slist)])
    fvibstd = np.std(flist[len(flist)-n_last:len(flist)])
    svibstd = np.std(slist[len(slist)-n_last:len(slist)])
    if has_neg:
        print("Converged [NEGATIVE] (K,kJ/mol,J/K/mol): T = %.2f fvib = %.3f fvibstd = %.5f svib = %.3f svibstd = %.5f" % (t,fvib,fvibstd,svib,svibstd),flush=True)
        print("%.2f %.8f %.8f %.8f %.8f (NEG)" % (t,fvib,fvibstd,svib,svibstd),file=fout,flush=True)
    else:
        print("Converged (K,kJ/mol,J/K/mol): T = %.2f fvib = %.3f fvibstd = %.5f svib = %.3f svibstd = %.5f" % (t,fvib,fvibstd,svib,svibstd),flush=True)
        print("%.2f %.8f %.8f %.8f %.8f" % (t,fvib,fvibstd,svib,svibstd),file=fout,flush=True)

    if write_fc2eff == True:
        fc2 = ForceConstants.from_arrays(scel, fc2_array=(fc2 / fc_factor),
                                         fc3_array=None)
        fc2.write_to_phonopy(f'./{prefix}-{t:04d}.fc2_eff', format='hdf5')

fout.close()

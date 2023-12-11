## 3-calculate_effective_frequencies.py: calculate effective frequencies from FCn
##
## Input: prefix.info, prefix.cs, prefix.fcn
## -> optional: prefix.fc2_lr ## subtract LR from reference forces first, if file present
## Output: prefix.svib temp.fc2_eff
## EB last update: stop if imaginary frequencies appear

import os
import time
import pickle
import numpy as np

## input block ##
prefix="blh" ## prefix for the generated files
n_structures = 10 # number of structures used in scph
train_fraction=0.8 # fraction of data used in training/validation split
## n_iterations = 40 # iterations in scph, at least 5
fit_method="rfe" # training method
## temperatures = np.arange(440, 0, -10)# temperature list (0 is always included)
temperatures = [300, 3000]
## EB write and read fc2_eff
write = True # write temp.fc2_eff
born_file = 'BORN'## else None ## NOT necessary i think
##  EB Restart temperature 
restart_temperature = None
alpha_initial = 0.2

## EB new parameters
n_max = 300 # max number of steps
n_safe = 30 # requiered steps with real frequencies
n_dead = 20 # max consecutive steps with non-real frequencies
n_last = 20 # last steps considered for the fvib, svib, fvibstd and svibstd calculation


#################

from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.cutoffs import estimate_maximum_cutoff
from hiphive.calculators import ForceConstantCalculator
from hiphive.force_constant_model import ForceConstantModel
from hiphive.structure_generation import generate_rattled_structures, generate_phonon_rattled_structures
from hiphive.utilities import prepare_structures
## from phonopy import Phonopy, load
from trainstation import Optimizer
from phonopy.file_IO import parse_BORN

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
        ## return eV/Ang**2
        fc2_LR = fc2_LR * fc_factor
# build the cs for the schp calculation, force constants, and calculator
cs = ClusterSpace(cell,[estimate_maximum_cutoff(scel)-1e-4])
fcs = fcp.get_force_constants(scel)
calc = ForceConstantCalculator(fcs)

## EB already in phcal
# prepare the phonopy object for the fvib/svib calculation
## atoms_phonopy = PhonopyAtoms(symbols=cell.get_chemical_symbols(),
##                              scaled_positions=cell.get_scaled_positions(),
##                              cell=cell.cell)


## ph = Phonopy(atoms_phonopy, supercell_matrix=ncell*np.eye(3),
##              primitive_matrix=None,calculator=phcalc)

## EB already in phcel, but just in case
print(phcel.nac_params)
print(phcel.primitive)
if born_file is not None:
    phcel.nac_params = parse_BORN(phcel.primitive, symprec=1e-5, is_symmetry=True,
                                  filename=born_file)
print(phcel.nac_params)


# EB just in case an FC file is around
phcel.force_constants = np.zeros((len(scel), len(scel), 3, 3))

# open output file
fout = open(f'{prefix}.svib' ,"w")
## EB restart frquencies
if restart_temperature is not None:
    from hiphive import ForceConstants
    fc2 = ForceConstants.read_phonopy(supercell=scel,
            fname=f'{prefix}-{restart_temperature:04d}.fc2_eff',
            format='hdf5')
    ## EB return correct units
    fc2 = fc2.get_fc_array(order=2) * fc_factor

else:
    fc2 = fcp.get_force_constants(scel).get_fc_array(order=2)

if fc2_LR is not None:
    fc2 += fc2_LR

# calculate the harmonic quantities
## EB depreciated
## ph.set_force_constants(fc2 / fc_factor)
phcel.force_constants = fc2 / fc_factor  ## fc2 still in eV/ang**2
phcel.run_mesh([20] * 3)
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
rattled_structures = generate_rattled_structures(scel, n_structures, 0.15)
rattled_structures = prepare_structures(rattled_structures, scel, calc, check_permutation=False)
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
    ##EB test variable alpha 
    count = 0
    dead_counter = 0
    alpha = alpha_initial ## update parameter outside the loop
    for i in range(max(n_max,1)):

        # generate structures with new FC2, including the LR correction
        fcm.parameters = param_old
        fc2 = fcm.get_force_constants().get_fc_array(order=2)
        if fc2_LR is not None:
            fc2 += fc2_LR
        ## err:
        print(fc2[0][0][0] / fc_factor)
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

        ## EB update alpha parameter 
       ## if count % 10 == 0:
       ##     alpha = alpha - 0.1 ## anything


        # calculate fvib 
        param_new = alpha * opt.parameters + (1-alpha) * param_old
        ## son sr solo
        fc2 = ForceConstantPotential(cs, param_new).get_force_constants(scel).get_fc_array(order=2)

        if fc2_LR is not None:
            fc2 += fc2_LR
        ## 
        phcel.force_constants = fc2 / fc_factor  ## fc2 still in eV/ang**2
        phcel.run_mesh([20] * 3)
        phcel.run_thermal_properties(temperatures=[t])
        ## #EB to remove
        ## phcel.auto_band_structure(write_yaml=True)

        ## EB before printing
        x_new_norm = np.linalg.norm(param_new)
        delta_x_norm = np.linalg.norm(param_old-param_new)

        ## EB stop block
        fvib = phcel.get_thermal_properties_dict()['free_energy'][0]
        if np.isnan(fvib):
            ## print('imaginary freqs')
            ## initial parameter must change at least randomly
            ## alpha muy grande tambien puede estar bien y un alpha menor/ o 2 alpha
            ## y otor alpha medios
            dead_counter += 1 
            ## EB
            ## rand = np.random.rand() 
            rand = 0.4
            param_old = rand * opt.parameters + (1-rand)*param_old

        if not np.isnan(fvib):
            ## print('only real')
            if count == n_dead // 2:
                dead_counter = 0
            svib = phcel.get_thermal_properties_dict()['entropy'][0]
            flist.append(fvib)
            slist.append(svib)
            ## EB param last can be removed
            param_last = param_new
            param_old = param_new
            count += 1
        ## EB print iteration summary
        disps = [atoms.get_array('displacements') for atoms in phonon_rattled_structures]
        disp_ave = np.mean(np.abs(disps))
        disp_max = np.max(np.abs(disps))
        ## EB alpha included ##        
        print(f'{i}: x_new = {x_new_norm:.3e}, delta_x = {delta_x_norm:.3e},', 
              f'disp_ave = {disp_ave:.5f}, fvib = {fvib:.3f},',
              f'svib = {svib:.3f}, alpha = {alpha:.2f}')

##        ## EB get G (gamma) and R freqs
##        freqs = ph.get_frequencies([0, 0, 0]) ## frequencies only get_frequencies_with_eigenvectors()
##        print(f'freq G: {freqs[0]:.5f} {freqs[1]:.5f} {freqs[2]:.5f} {freqs[3]:.5f} {freqs[4]:.5f}') 
##        freqs = ph.get_frequencies([0.5, 0.5, 0.5]) ## frequencies only get_frequencies_with_eigenvectors()
##        print(f'freq R: {freqs[0]:.5f} {freqs[1]:.5f} {freqs[2]:.5f} {freqs[3]:.5f} {freqs[4]:.5f}') 
        ## EB restart counter if n_dead/2 consecutive bad steps 
        if dead_counter == n_dead//2:
            count = 0

        if dead_counter == n_dead:
            print(f'Frequencies not converged at {t} loop ended') 
            exit()

        if i == n_max-1:
            print(f'Frequencies not converged at {t} loop ended') 
            exit()

        if count == n_safe:
            ## EB print last values only
            fvib = np.mean(flist[len(flist)-n_last:len(flist)])
            svib = np.mean(slist[len(slist)-n_last:len(slist)])
            fvibstd = np.std(flist[len(flist)-n_last:len(flist)])
            svibstd = np.std(slist[len(slist)-n_last:len(slist)])
            break

        ## Update params
        # print iteration summary
        

    # standard deviation of fvib and svib and final message
    ## average not only last
    print("Converged (K,kJ/mol,J/K/mol): T = %.2f fvib = %.3f fvibstd = %.5f svib = %.3f svibstd = %.5f" % (t,fvib,fvibstd,svib,svibstd))
    print("%.2f %.8f %.8f %.8f %.8f" % (t,fvib,fvibstd,svib,svibstd),file=fout)
    ## Write fc2_eff EB using the corresponing units
    if write == True:
        from hiphive import ForceConstants
        fc2 = ForceConstants.from_arrays(scel, fc2_array=(fc2 / fc_factor), fc3_array=None)
        fc2.write_to_phonopy(f'./{prefix}-{t:04d}.fc2_eff', format='hdf5')
fout.close()

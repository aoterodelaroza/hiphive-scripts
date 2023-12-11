import pickle
import os
from hiphive import ForceConstantPotential
from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5
import numpy as np

prefix = 'srtio3'
with open(prefix + ".info","rb") as f:
    calculator, phcalc, ncell, cell, scel, fc_factor, phcel = pickle.load(f)

if os.path.isfile(prefix + ".fc2_lr"):
    with open(prefix + ".fc2_lr","rb") as f:
        ## fc2_LR written using the calculator units 
        fc2_LR = pickle.load(f) * fc_factor

fcp = ForceConstantPotential.read(f'{prefix}.fcn')
fc2 = fcp.get_force_constants(scel).get_fc_array(order=2) 
if os.path.isfile(prefix + ".fc2_lr"):
    fc2 += fc2_LR

fc2 = fc2 / fc_factor
write_FORCE_CONSTANTS(fc2) 

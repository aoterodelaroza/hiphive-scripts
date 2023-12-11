import pickle
from hiphive import ForceConstants
from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5

prefix = 'cao'

with open(prefix + ".fc2_harmonic","rb") as f:
    fc2 = pickle.load(f)
    fc2 = fc2 


## fc2 = ForceConstants.from_arrays(scel, fc2_array=(fc2), fc3_array=None)

write_FORCE_CONSTANTS(fc2) 
## fc2.write_to_phonopy(f'./{prefix}-{t:04d}.fc2_eff', format='hdf5')

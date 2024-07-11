#!/usr/bin/env python

## Usage: python make_born_q2r.py NaCl.in NaCl.fc > BORN

## From the phonopy distribution.

import sys
import numpy as np
from phonopy.structure.symmetry import elaborate_borns_and_epsilon
from phonopy.interface.qe import read_pwscf, PH_Q2R

primcell_filename = sys.argv[1]
q2r_filename = sys.argv[2]
cell, _ = read_pwscf(primcell_filename)
q2r = PH_Q2R(q2r_filename)
q2r.run(cell, parse_fc=False)
if q2r.epsilon is not None:
    borns, epsilon, _ = elaborate_borns_and_epsilon(
        cell,
        q2r.borns,
        q2r.epsilon,
        supercell_matrix=np.diag(q2r.dimension),
        symmetrize_tensors=True)
    print("2.000000")
    print(("%13.8f" * 9) % tuple(epsilon.ravel()))
    for z in borns:
        print(("%13.8f" * 9) % tuple(z.ravel()))

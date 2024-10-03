"""
This module contains new support/utility functions.
"""

from glob import glob
import numpy as np
from hiphive import StructureContainer
from hiphive.utilities import get_displacements
import ase, ase.units
import multiprocessing as mp

def constant_rattle(atoms, n_structures, amplitude, seed=None):
    """
    Generate n_structures strucures based on the initial structure
    atoms by rattling the atomic positions randomly with displacements
    given by amplitude.  If seed is RandomState, use the random
    generator, otherwise initialize.
    """
    if seed is None:
        import time
        seed = int(time.time())
    if type(seed) is np.random.RandomState:
        rs = seed
    else:
        rs = np.random.RandomState(seed)

    ## number of atoms
    N = len(atoms)

    ## list for new structures
    atoms_list = []
    for _ in range(n_structures):
        atoms_tmp = atoms.copy()
        rand_dir = rs.randn(3, N)
        norm = np.sqrt((rand_dir**2).sum(axis=0))
        rand_dir = (rand_dir / norm).T
        atoms_tmp.positions += rand_dir * amplitude * 1/3
        atoms_list.append(atoms_tmp)

    return atoms_list

def least_squares(M, F, skiprmse=None):
    """
    Run least squares with matrices M and F and returns the
    least-squares coefficients. If skiprmse is None,
    also return the root mean square error, the average absolute F,
    the r2 coefficient and the adjusted r2.
    """

    coefs = np.linalg.solve(M.T.dot(M),M.T.dot(F))
    if skiprmse is not None:
        return coefs, 0., 0., 0., 0.

    nparam = M.shape[1]
    Fnum = len(F)
    ssq = np.sum((M.dot(coefs) - F)**2)
    sstot = np.sum((F - np.mean(F))**2)

    rmse = np.sqrt(ssq / Fnum)
    r2 = 1 - ssq / sstot
    ar2 = 1 - (1 - r2) * (Fnum - 1) / (Fnum - nparam - 1)
    Fabsavg = np.sum(np.abs(F)) / Fnum

    return coefs, rmse, Fabsavg, r2, ar2

def thread_init(scel_,cs_,fc2_LR_,coefs_,Fmean_):
    """
    Helper routine for thread initialization in least_squares_batch
    """
    global scel, cs, fc2_LR, coefs, Fmean
    scel = scel_
    cs = cs_
    fc2_LR = fc2_LR_
    coefs = coefs_
    Fmean = Fmean_

def thread_task(fname):
    """
    Helper routine for thread work in least_squares_batch
    """
    global scel, cs, fc2_LR, coefs, Fmean

    # read structure
    atoms = ase.io.read(fname)

    # this is because otherwise the atoms are not in POSCAR order
    displacements = get_displacements(atoms, scel)
    forces = atoms.get_forces()

    # append to the structure container
    atoms_tmp = scel.copy()
    atoms_tmp.new_array('displacements', displacements)
    atoms_tmp.new_array('forces', forces)

    # create the structure container
    sc = StructureContainer(cs)
    sc.add_structure(atoms_tmp)

    # create M and F
    if fc2_LR is not None:
        displacements = np.array([fs.displacements for fs in sc])
        M, F = sc.get_fit_data()
        F -= np.einsum('ijab,njb->nia', -fc2_LR, displacements).flatten()
    else:
        M, F = sc.get_fit_data()


    if coefs is None:
        ### generate the A and b contributions (first pass)
        Asum = M.T.dot(M)
        bsum = M.T.dot(F)

        # print message
        print("[%d] %s %4d %10.4f %10.4f %10.4f" % (mp.current_process().pid,fname,len(sc[0]),
                                                    np.mean([np.linalg.norm(d) for d in sc[0].displacements]),
                                                    np.mean([np.linalg.norm(d) for d in sc[0].forces]),
                                                    np.max([np.linalg.norm(d) for d in sc[0].forces])))

        return Asum, bsum, np.sum(np.abs(F)), np.sum(F), len(F)
    else:
        ### calculate ssq and sstot (second pass)
        ssq = np.sum((M.dot(coefs) - F)**2)
        sstot = np.sum((F - Fmean)**2)
        return ssq, sstot

def least_squares_batch(structs,nthread,cs=None,scel=None,fc2_LR=None,skiprmse=None):
    """
    Run least squares in batches to preserve memory using the M and F
    values from the structures in structs. structs can be:

    - A string/regexp or a list of strings/regexps. Requires giving
    the clusterspace (cs) and supercell structure (scel).

    - An ASE Atoms object or a list of Atoms objects. Requires the
    clusterspace (cs).

    - A structure container. In this case, the effect is the same as
    least_squares_accum.

    This version takes longer than least_squares_accum but uses less memory.
    """

    ## special case: structs is a structure container => use simple least squares w all data
    if (isinstance(structs,StructureContainer)):
        if fc2_LR is not None:
            displacements = np.array([fs.displacements for fs in structs])
            M, F = structs.get_fit_data()
            F -= np.einsum('ijab,njb->nia', -fc2_LR, displacements).flatten()
        else:
            M, F = structs.get_fit_data()
        return least_squares(M,F,skiprmse)

    ## initialize file list and matrices
    print("\n## least_squares_batch ##")

    ## build the file lists
    lfile = []
    if isinstance(structs,str):
        lfile.extend(glob(structs))
    else:
        for i in structs:
            if isinstance(i,str):
                lfile.extend(glob(i))
            else:
                lfile.append(i)

    ## initialize
    Fsum = 0.
    Fsumabs = 0.
    Fnum = 0.
    nparam = cs.n_dofs
    A = np.zeros((nparam, nparam))
    b = np.zeros((nparam,))

    ## calculate the least-squares matrices (and other data) in parallelized batches
    print("## calculating A,b matrices (parallel)")
    print("#[pid] structure-name num-atoms avg-disp avg-force max-force")
    pool = mp.pool.Pool(nthread,initializer=thread_init,initargs=(scel,cs,fc2_LR,None,None))
    for result in pool.imap_unordered(thread_task,lfile):
        A += result[0]
        b += result[1]
        Fsumabs += result[2]
        Fsum += result[3]
        Fnum += result[4]
    Fmean = Fsum / Fnum

    ## run the least squares to calculate coefficients
    print("## running least squares")
    coefs = np.linalg.solve(A.T.dot(A),A.T.dot(b))
    del A,b

    if skiprmse is None:
        ## header and initialize
        print("## calculating rmse (parallel)")
        ssq = 0.
        sstot = 0.

        ## calculate the rmse in parallel
        pool = mp.pool.Pool(nthread,initializer=thread_init,initargs=(scel,cs,fc2_LR,coefs,Fmean))
        for result in pool.imap_unordered(thread_task,lfile):
            ssq += result[0]
            sstot += result[1]

        ## calculate rmse, r2, adjusted r2
        rmse = np.sqrt(ssq / Fnum)
        r2 = 1 - ssq / sstot
        ar2 = 1 - (1 - r2) * (Fnum - 1) / (Fnum - nparam - 1)
    else:
        rmse = 0.
        r2 = 0.
        ar2 = 0.
    Fabsavg = Fsumabs/Fnum
    print()

    return coefs, rmse, Fabsavg, r2, ar2

def least_squares_accum(structs,cs=None,scel=None,fc2_LR=None,skiprmse=None):
    """
    Run least squares using the M and F values from the structures in
    structs. structs can be:

    - A string/regexp or a list of strings/regexps. Requires giving
    the clusterspace (cs) and supercell structure (scel).

    - An ASE Atoms object or a list of Atoms objects. Requires the
    clusterspace (cs).

    - A structure container. In this case, the effect is the same as
    least_squares_accum.

    This version is faster than least_squares_accum but loads the
    whole M into memory.
    """

    ## build the structure container
    if (isinstance(structs,StructureContainer)):
        sc = structs
    else:
        ## build our own sc
        sc = StructureContainer(cs)

        ## build the file lists
        lfile = []
        if isinstance(structs,str):
            lfile.extend(glob(structs))
        else:
            for i in structs:
                if isinstance(i,str):
                    lfile.extend(glob(i))
                else:
                    sc.add_structure(i)

        ## must be strings with filenames
        if lfile:
            print("\n## least_squares_accum ##")
            for fname in lfile:
                atoms = ase.io.read(fname)

                # this is because otherwise the atoms are not in POSCAR order
                displacements = get_displacements(atoms, scel)
                forces = atoms.get_forces()

                # append to the structure container
                atoms_tmp = scel.copy()
                atoms_tmp.new_array('displacements', displacements)
                atoms_tmp.new_array('forces', forces)
                sc.add_structure(atoms_tmp)
                print("%s %4d %10.4f %10.4f %10.4f" % (fname,len(sc[0]),np.mean([np.linalg.norm(d) for d in sc[0].displacements]),
                                                       np.mean([np.linalg.norm(d) for d in sc[0].forces]),
                                                       np.max([np.linalg.norm(d) for d in sc[0].forces])))

    if fc2_LR is not None:
        displacements = np.array([fs.displacements for fs in sc])
        M, F = sc.get_fit_data()
        F -= np.einsum('ijab,njb->nia', -fc2_LR, displacements).flatten()
    else:
        M, F = sc.get_fit_data()
    return least_squares(M,F,skiprmse)

def has_negative_frequencies(freqs,threshold=10):
    """
    Return whether there are negative frequencies in the input
    mesh. Criterion: the average of the negative frequencies in
    absolute value and cm-1 is above threshold.
    """
    from phonopy.units import THzToCm

    mask = freqs < 0
    nneg = np.sum(mask)
    return (nneg > 0) and (sum(abs(freqs[mask])) * THzToCm / nneg > threshold)

def write_negative_frequencies_file(mesh,filename):
    """
    Given a phonopy mesh, write the list of mesh points where there
    are negative frequencies to the file filename.
    """
    from phonopy.units import THzToCm
    print("Negative frequencies file written to: ",filename)

    f = open(filename,"w")
    print("# List of negative frequencies and q-points",file=f)
    iqlist, ifreqlist = np.where(mesh.frequencies < 0)

    for idx, (iq,ifreq) in enumerate(zip(iqlist,ifreqlist)):
        print("%10.3f   %10.7f, %10.7f, %10.7f" % (
            mesh.frequencies[iq][ifreq]*THzToCm,mesh._qpoints[iq][0],
            mesh._qpoints[iq][1],mesh._qpoints[iq][2]),file=f)
    f.close()

#### TEMPORARY: this is a copy of the phonon rattle code in hiphive.structure_generation ####
#### modified to handle negative frequencies better (set minimum frequency) ####

def _n_BE(T, w_s):
    """
    Bose-Einstein distribution function.

    Parameters
    ---------
    T : float
        Temperature in Kelvin
    w_s: numpy.ndarray
        frequencies in eV (3*N,)

    Returns
    ------
    Bose-Einstein distribution for each energy at a given temperature
    """

    with np.errstate(divide='raise', over='raise'):
        try:
            n = 1 / (np.exp(w_s / (ase.units.kB * T)) - 1)
        except Exception:
            n = np.zeros_like(w_s)
    return n


def _phonon_rattle(m_a, T, w2_s, e_sai, QM_statistics):
    """ Thermal excitation of phonon modes as described by West and
    Estreicher, Physical Review Letters  **96**, 115504 (2006).

    _s is a mode index
    _i is a Carteesian index
    _a is an atom index

    Parameters
    ----------
    m_a : numpy.ndarray
        masses (N,)
    T : float
        temperature in Kelvin
    w2_s : numpy.ndarray
        the squared frequencies from the eigenvalue problem (3*N,)
    e_sai : numpy.ndarray
        polarizations (3*N, N, 3)
    QM_statistics : bool
        if the amplitude of the quantum harmonic oscillator shoule be used
        instead of the classical amplitude

    Returns
    -------
    displacements : numpy.ndarray
        shape (N, 3)
    """
    n_modes = 3 * len(m_a)

    # skip 3 translational modes
    argsort = np.argsort(np.abs(w2_s))
    e_sai = e_sai[argsort][3:]
    w2_s = w2_s[argsort][3:]

    w_s = np.sqrt(np.abs(w2_s))

    #### xxxx ####
    imag_mask = w_s < 0.1
    w_s[imag_mask] = 0.1

    prefactor_a = np.sqrt(1 / m_a).reshape(-1, 1)
    if QM_statistics:
        hbar = ase.units._hbar * ase.units.J * ase.units.s
        frequencyfactor_s = np.sqrt(hbar * (0.5 + _n_BE(T, hbar * w_s)) / w_s)
    else:
        frequencyfactor_s = 1 / w_s
        prefactor_a *= np.sqrt(ase.units.kB * T)

    phases_s = np.random.uniform(0, 2 * np.pi, size=n_modes - 3)
    amplitudes_s = np.sqrt(-2 * np.log(1 - np.random.random(n_modes - 3)))

    u_ai = prefactor_a * np.tensordot(
            amplitudes_s * np.cos(phases_s) * frequencyfactor_s, e_sai, (0, 0))
    return u_ai  # displacements


class _PhononRattler:
    """
    Class to be able to conveniently save modes and frequencies needed
    for phonon rattle.

    Parameters
    ----------
    masses : numpy.ndarray
        masses (N,)
    force_constants : numpy.ndarray
        second order force constant matrix, with shape `(3N, 3N)` or
        `(N, N, 3, 3)`. The conversion will be done internally if.
    imag_freq_factor: float
        If a squared frequency, w2, is negative then it is set to
        w2 = imag_freq_factor * np.abs(w2)
    """
    def __init__(self, masses, force_constants, imag_freq_factor=1.0):
        n_atoms = len(masses)
        if len(force_constants.shape) == 4:  # assume shape = (n_atoms, n_atoms, 3, 3)
            force_constants = force_constants.transpose(0, 2, 1, 3)
            force_constants = force_constants.reshape(3 * n_atoms, 3 * n_atoms)
            # Now the fc should have shape = (n_atoms * 3, n_atoms * 3)
        # Construct the dynamical matrix
        inv_root_masses = (1 / np.sqrt(masses)).repeat(3)
        D = np.outer(inv_root_masses, inv_root_masses)
        D *= force_constants
        # find frequnecies and eigenvectors
        w2_s, e_sai = np.linalg.eigh(D)
        # reshape to get atom index and Cartesian index separate
        e_sai = e_sai.T.reshape(-1, n_atoms, 3)

        # The three modes closest to zero are assumed to be zero, ie acoustic sum rules are assumed
        frequency_tol = 1e-6
        argsort = np.argsort(np.abs(w2_s))
        w2_gamma = w2_s[argsort][:3]

        # treat imaginary modes as real
        imag_mask = w2_s < -frequency_tol
        w2_s[imag_mask] = imag_freq_factor * np.abs(w2_s[imag_mask])

        self.w2_s = w2_s
        self.e_sai = e_sai
        self.masses = masses

    def __call__(self, atoms, T, QM_statistics):
        """ rattle atoms by adding displacements

        Parameters
        ----------
        atoms : ase.Atoms
            Ideal structure to add displacements to.
        T : float
            temperature in Kelvin
        """
        u_ai = _phonon_rattle(self.masses, T, self.w2_s, self.e_sai,
                              QM_statistics)
        atoms.positions += u_ai

def generate_phonon_rattled_structures(atoms, fc2, n_structures, temperature,
                                       QM_statistics=False, imag_freq_factor=1.0):

    structures = []
    pr = _PhononRattler(atoms.get_masses(), fc2, imag_freq_factor)
    for _ in range(n_structures):
        atoms_tmp = atoms.copy()
        pr(atoms_tmp, temperature, QM_statistics)
        structures.append(atoms_tmp)
    return structures

## already in structure generation

"""
This module contains new support/utility functions.
"""

from glob import glob
import numpy as np
from hiphive import StructureContainer
from hiphive.utilities import get_displacements
import ase

def constant_rattle(atoms, n_structures, amplitude, seed=None):
    """
    Generate n_structures based on the initial structure atoms by rattling
    the atomic positions randomly with displacements given by amplitude.
    If seed is RandomState, use the random generator, otherwise initialize.
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

def least_squares(M, F, n_jobs=-1, verbose=1, standardize=True, mean=False,
                  std=True, fout=None):
    """
    Old version of least_squares = TO BE REMOVED.

    StandardScaler:
        s = (x -u)/s , u = mean(x), s = std(x)
    Return the model and the parameters
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
    import time
    #M, F = sc.get_fit_data()
    opt = LinearRegression(n_jobs=n_jobs)
    if standardize:
        from sklearn.preprocessing import StandardScaler
        s = StandardScaler(copy=False, with_mean=mean, with_std=std)
        s.fit_transform(M)
        factor = 1.0 / np.std(F)
        F = F * factor
        a = time.time()
        opt.fit(M, F)
        s.inverse_transform(M)
        parameters = opt.coef_ / factor
        s.transform(parameters.reshape(1, -1)).reshape(-1,) ## restore par.
        opt.coef_ = parameters
        #original F
        F = F / factor
    else:
        opt.fit(M, F)
        factor = 1

    y_pred = opt.predict(M)
    rmse = np.sqrt(mean_squared_error(F, y_pred))
    if verbose > 0:
        print(f"""====================================================
Parameters: {opt.n_features_in_}
Non-zero parameters (>1e-3): {len(np.where(np.abs(opt.coef_) > 1e-3)[0])}
R2: {opt.score(M, F):0.8f}
Mean Absolute Error: {mean_absolute_error(F, y_pred):0.8f}
Mean Squared Error: {mean_squared_error(F, y_pred):0.8f}
Max Error: {max_error(F, y_pred):0.8f}
====================================================
              """, file=fout)
    return opt, opt.coef_, rmse

def least_squares_simple(M, F):
    """
    Run a simple and efficient version of least squares and return the
    least-squares parameters and the root mean square error.
    """

    coefs = np.linalg.solve(M.T.dot(M),M.T.dot(F))
    rmse = np.sqrt(np.linalg.norm(M.dot(coefs) - F)**2 / len(F))

    return coefs, rmse

def least_squares_batch_simple(outputs,cs,scel,fc2_LR=None):
    """
    Least squares, in batches
    """

    ## initialize file list and matrices
    print("\n## least_squares_batch_simple ##")
    lfile = glob(outputs)
    A = None
    b = None
    Fsum = 0
    Fnum = 0

    # read the forces and build the structure container
    print("# name num-atoms avg-disp avg-force max-force")
    for fname in glob(outputs):
        ## read the structure and fill a structure container with just one strucutre
        sc = StructureContainer(cs)
        atoms = ase.io.read(fname)

        ## initialize A and b if not done already
        if A is None or b is None:
            A = np.zeros((cs.n_dofs, cs.n_dofs))
            b = np.zeros((cs.n_dofs,))

        # this is because otherwise the atoms are not in POSCAR order
        displacements = get_displacements(atoms, scel)
        forces = atoms.get_forces()

        # append to the structure container
        atoms_tmp = scel.copy()
        atoms_tmp.new_array('displacements', displacements)
        atoms_tmp.new_array('forces', forces)
        sc.add_structure(atoms_tmp)

        print("%s %4d %10.4f %10.4f %10.4f" % (fname,len(sc[0]),\
              np.mean([np.linalg.norm(d) for d in sc[0].displacements]),np.mean([np.linalg.norm(d) for d in sc[0].forces]),\
              np.max([np.linalg.norm(d) for d in sc[0].forces])))

        if fc2_LR is not None:
            displacements = np.array([fs.displacements for fs in sc])
            M, F = sc.get_fit_data()
            F -= np.einsum('ijab,njb->nia', -fc2_LR, displacements).flatten()
        else:
            M, F = sc.get_fit_data()

        A += M.T.dot(M)
        b += M.T.dot(F)
        Fsum += np.sum(np.abs(F))
        Fnum += len(F)

    ## run the least squares to calculate coefficients
    print("## running least squares")
    del M,F,sc,atoms,displacements,forces,atoms_tmp
    coefs = np.linalg.solve(A.T.dot(A),A.T.dot(b))
    del A,b

    ## calculate rmse
    print("## calculating rmse")
    rmse = 0.
    # read the forces and build the structure container
    for fname in glob(outputs):
        ## read the structure and fill a structure container with just one strucutre
        sc = StructureContainer(cs)
        atoms = ase.io.read(fname)

        # this is because otherwise the atoms are not in POSCAR order
        displacements = get_displacements(atoms, scel)
        forces = atoms.get_forces()

        # append to the structure container
        atoms_tmp = scel.copy()
        atoms_tmp.new_array('displacements', displacements)
        atoms_tmp.new_array('forces', forces)
        sc.add_structure(atoms_tmp)

        if fc2_LR is not None:
            displacements = np.array([fs.displacements for fs in sc])
            M, F = sc.get_fit_data()
            F -= np.einsum('ijab,njb->nia', -fc2_LR, displacements).flatten()
        else:
            M, F = sc.get_fit_data()

        rmse += np.sum((M.dot(coefs) - F)**2)
    del M,F,sc,atoms,displacements,forces,atoms_tmp
    rmse = np.sqrt(rmse / Fnum)
    print()

    return coefs, rmse, Fsum/Fnum*1000

def shuffle_split_cv(M, F, n_splits=5, test_size=0.2, seed=None, verbose=1,
                     standardize=True, last=False, fout=None):
    """
    Shuffle-splict cross validation using least squares = TO BE REMOVED.

    Standard ShuffleSplit cross validation
    KFold is not available for our case as in trainstation
    method only available least-squares add if if more fitting methods are add
    """
    from sklearn.model_selection import ShuffleSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
    if seed is None:
        import time
        seed = int(time.time())
    if type(seed) is np.random.RandomState:
        rs = seed
    else:
        rs = np.random.RandomState(seed)

    sp = ShuffleSplit(n_splits=n_splits, test_size=test_size,
                      train_size=1-test_size, random_state=rs)
    parameters = []
    scores = []
    maes = []
    rmses = []
    maxs = []
    for index, (train, test) in enumerate(sp.split(M)):
        #opt.fit(M[train], F[train])
        opt, parameter, _ = least_squares(M[train], F[train], standardize=standardize,
                                          verbose=0)
        y_train = opt.predict(M[train])
        y_test = opt.predict(M[test])
        scores.append([opt.score(M[train], F[train]), opt.score(M[test],
                                                                F[test])])
        maes.append([mean_absolute_error(F[train], y_train),
                     mean_absolute_error(F[test], y_test)])

        rmses.append([mean_squared_error(F[train], y_train),
                      mean_squared_error(F[test], y_test)])
        maxs.append([max_error(F[train], y_train), max_error(F[test], y_test)])
        parameters.append(parameter)
    if last:
        ## uses final split parameters for final
        opt.coef_ = parameter
    else:
        ## mean value of all the splits
        opt.coef_ = np.mean(parameters, axis=0)
    if verbose > 1:
        print_cv_steps(n_splits, scores, maes, rmses, maxs)

    y_pred = opt.predict(M)
    if verbose > 0:
        print(f"""====================================================
Parameters: {opt.n_features_in_}
Non-zero parameters (>1e-3): {len(np.where(np.abs(opt.coef_) > 1e-3)[0])}
TRAINING SET:
R2 train: {np.mean(np.array(scores),axis=0)[0]:0.8f}
Mean Absolute Error train: {np.mean(np.array(maes),axis=0)[0]:0.8f}
Mean Squared Error train: {np.mean(np.array(rmses),axis=0)[0]:0.8f}
Max Error train: {np.mean(np.array(maxs),axis=0)[0]:0.8f}
TEST SET:
R2 test: {np.mean(np.array(scores),axis=0)[1]:0.8f}
Mean Absolute Error test: {np.mean(np.array(maes),axis=0)[1]:0.8f}
Mean Squared Error test: {np.mean(np.array(rmses),axis=0)[1]:0.8f}
Max Error test: {np.mean(np.array(maxs),axis=0)[1]:0.8f}
FINAL MODEL:
R2 final: {opt.score(M, F):0.8f}
Mean Absolute Error final: {mean_absolute_error(F, y_pred):0.8f}
Mean Squared Error final: {mean_squared_error(F, y_pred):0.8f}
Max Error final: {max_error(F, y_pred):0.8f}
====================================================
                  """, file=fout)
    return opt, opt.coef_, np.sqrt(mean_squared_error(F, y_pred))

def print_cv_steps(splits, scores, maes, rmses, maxs, fout=None):
    """
    Just to print the cross validation metrics of each step
    TO BE REMOVED.
    """
    for i, score, mae, rmse, max in zip(range(splits), scores, maes, rmses, maxs):
        print(f"""============================
Fold: {i+1}
R2 train: {score[0]:0.8f}
R2 test: {score[1]:0.8f}
MAE train: {mae[0]:0.8f}
MAE test: {mae[1]:0.8f}
RMSE train: {rmse[0]:0.8f}
RMSE test: {rmse[1]:0.8f}
Max Error train: {max[0]:0.8f}
Max Error test: {max[1]:0.8f}
============================""", file=fout)

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
            n = 1 / (np.exp(w_s / (aunits.kB * T)) - 1)
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
        hbar = aunits._hbar * aunits.J * aunits.s
        frequencyfactor_s = np.sqrt(hbar * (0.5 + _n_BE(T, hbar * w_s)) / w_s)
    else:
        frequencyfactor_s = 1 / w_s
        prefactor_a *= np.sqrt(aunits.kB * T)

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

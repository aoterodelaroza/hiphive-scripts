## already in structure generation
"""
This module contains new support/utility functions.
"""
import numpy as np
def constant_rattle(atoms, n_structures, amplitude, seed=None):
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
        # size (N, 3)
        rand_dir = (rand_dir / norm).T
        #print(rand_dir * amplitude)
        atoms_tmp.positions += rand_dir * amplitude * 1/3
        atoms_list.append(atoms_tmp)

    return atoms_list

def least_squares(M, F, n_jobs=-1, verbose=1, standardize=True, mean=False,
                  std=True, fout=None):
    """
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

def shuffle_split_cv(M, F, n_splits=5, test_size=0.2, seed=None, verbose=1,
                     standardize=True, last=False, fout=None):
    """
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

def least_squares_lapack(M, F, verbose=1, standardize=True,
                         mean=False, std=True, fout=None):
    """
    StandardScaler:
        s = (x -u)/s , u = mean(x), s = std(x)
    Return the model and the parameters
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
    from scipy.linalg import lstsq
    from sklearn.preprocessing import StandardScaler
    import time

    print('start scaler')
    s = StandardScaler(copy=False, with_mean=mean, with_std=std)
    print('end scaler')
    s.fit_transform(M)
    factor = 1.0 / np.std(F)
    F = F * factor
    a = time.time()
    print('star fit')
    coefs, _, _, _ = lstsq(M, F, lapack_driver='gelsd')
    print(f'end fit {time.time() - a}')
    s.inverse_transform(M)
    parameters = coefs / factor
    s.transform(parameters.reshape(1, -1)).reshape(-1,) ## restore par.
    coefs = parameters
    #original F
    F = F / factor
    print('end')

##     y_pred = opt.predict(M)
##     if verbose > 0:
##         print(f"""====================================================
## Parameters: {opt.n_features_in_}
## Non-zero parameters (>1e-3): {len(np.where(np.abs(opt.coef_) > 1e-3)[0])}
## R2: {opt.score(M, F):0.8f}
## Mean Absolute Error: {mean_absolute_error(F, y_pred):0.8f}
## Mean Squared Error: {mean_squared_error(F, y_pred):0.8f}
## Max Error: {max_error(F, y_pred):0.8f}
## ====================================================
##               """, file=fout)
    return 1, coefs


def numpy_fit(M, F):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
    from sklearn.preprocessing import StandardScaler
    from numpy.linalg import lstsq
    import time

    print('start scaler')
    s = StandardScaler(copy=False, with_mean=False, with_std=True)
    print('end scaler')
    s.fit_transform(M)
    factor = 1.0 / np.std(F)
    F = F * factor
    print(F.shape())
    a = time.time()
    print('star fit')
    coefs, _, _, _ = lstsq(M, F)
    print(f'end fit {time.time() - a}')
    s.inverse_transform(M)
    parameters = coefs / factor
    s.transform(parameters.reshape(1, -1)).reshape(-1,) ## restore par.
    coefs = parameters
    #original F
    F = F / factor
    print('end')
    return 1, coefs

def write_negative_frequencies_file(mesh,filename):
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


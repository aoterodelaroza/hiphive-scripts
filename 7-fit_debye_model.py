## 6-fit_debye_model.py: fit extended debye model to S(T) data and obtain
## fit parameters and list of (T,F,S,Cv)
##
## Input: prefix.info, prefix.svib
## Output: prefix.xdebye, prefix.thermal-data, prefix.png

import numpy as np

## input block ##
prefix="urea" ## prefix for the generated files
temperatures = np.arange(0, 300, 5) # extended temperature list
npoly_debye=3 # number of parameters in the polynomial part of extended Debye
neinstein=1 # number of Einstein terms
#################

import pickle
import scipy
import scipy.constants
import matplotlib.pyplot as plt
from pygsl.testing.sf import debye_3 as D3
from sklearn.metrics import r2_score

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, nthread_batch_lsqr, phcalc, ncell, cell, cell_for_cs, scel, fc_factor,\
        phcel, out_kwargs = pickle.load(f)
z = len(cell.get_chemical_symbols()) / len(phcel.primitive.masses)
kB = scipy.constants.k / scipy.constants.physical_constants['hartree-joule relationship'][0] # Ha/K
natom = len(cell)

# debye functions
def fdebye(t,thetad):
    x = thetad / t
    return -natom * kB * t * D3(x) + 3 * natom * kB * t * np.log(1 - np.exp(-x))

def sdebye(t,thetad):
    x = thetad / t
    return -3 * natom * kB * np.log(1 - np.exp(-x)) + 4 * natom * kB * D3(x)

def cvdebye(t,thetad):
    x = thetad / t
    return 12 * natom * kB * D3(x) - 9 * natom * kB * x / (np.exp(x) - 1)

# debye extended functions
def fdebye_ext(t,pin):
    thetad = pin[0]

    t = np.asarray(t)
    scalar_input = False
    if t.ndim == 0:
        t = t[None]
        scalar_input = True

    x = t / thetad
    f = np.zeros(x.size)
    idx = t > 1e-5
    f[idx] = - natom * kB * t[idx] * D3(1/x[idx]) + 3 * natom * kB * t[idx] * np.log(1 - np.exp(-1/x[idx]))

    term = 0.0
    for i in reversed([0] + [-i[1] / (i[0]+2) for i in enumerate(pin[1:])]):
        term = x * term + i
    f += natom * kB * t * term

    if scalar_input:
        return np.squeeze(f)
    return f

def sdebye_ext(t,pin):
    thetad = pin[0]

    t = np.asarray(t)
    scalar_input = False
    if t.ndim == 0:
        t = t[None]
        scalar_input = True

    x = t / thetad
    s = np.zeros(x.size)
    idx = t > 1e-5
    s[idx] = -3 * natom * kB * np.log(1 - np.exp(-1/x[idx])) + 4 * natom * kB * D3(1/x[idx])

    term = 0.0
    for i in reversed([0] + list(pin[1:])):
        term = x * term + i
    s += natom * kB * term
    if scalar_input:
        return np.squeeze(s)
    return s

def cvdebye_ext(t,pin):
    thetad = pin[0]

    t = np.asarray(t)
    scalar_input = False
    if t.ndim == 0:
        t = t[None]
        scalar_input = True

    x = t / thetad
    cv = np.zeros(x.size)
    idx = t > 1e-5
    cv[idx] = 12 * natom * kB * D3(1/x[idx]) - 9 * natom * kB / x[idx] / (np.exp(1/x[idx]) - 1)
    term = 0.0
    for i in reversed([i[1] * (i[0]+1) for i in enumerate(pin[1:])]):
        term = x * term + i
    cv += natom * kB * x * term

    if scalar_input:
        return np.squeeze(cv)
    return cv

# einstein functions
def feinstein(t,a):
    t = np.asarray(t)
    scalar_input = False
    if t.ndim == 0:
        t = t[None]
        scalar_input = True

    idx = t > 1e-5

    at = np.zeros(t.size)
    at[idx] = a/t[idx]
    emat = np.exp(-at)

    f = np.zeros(t.size)
    f[idx] = kB * t[idx] * np.log(1 - emat[idx])

    if scalar_input:
        return np.squeeze(f)
    return f

def seinstein(t,a):
    t = np.asarray(t)
    scalar_input = False
    if t.ndim == 0:
        t = t[None]
        scalar_input = True

    idx = t > 1e-5

    at = np.zeros(t.size)
    at[idx] = a/t[idx]
    emat = np.exp(-at)

    s = np.zeros(t.size)
    s[idx] = -kB * (np.log(1 - emat[idx]) - at[idx] * emat[idx] / (1-emat[idx]))

    if scalar_input:
        return np.squeeze(s)
    return s

def cveinstein(t,a):
    t = np.asarray(t)
    scalar_input = False
    if t.ndim == 0:
        t = t[None]
        scalar_input = True

    idx = t > 1e-5

    at = np.zeros(t.size)
    at[idx] = a/t[idx]
    emat1 = np.exp(at)

    cv = np.zeros(t.size)
    cv[idx] = kB * at[idx]**2 * emat1[idx] / (emat1[idx]-1)**2

    if scalar_input:
        return np.squeeze(cv)
    return cv

# combination functions
## comp = [npoly in debyeext, neinstein]
## pin = [coef_debyeext, td, -- npoly_debye --, coef_eins, a_eins, coef_eins, a_eins, ...]

def fcombine(t,pin,comp):
    if np.asarray(t).ndim == 0:
        f = 0
    else:
        f = np.zeros(t.size)

    ## extended debye
    f += pin[0] * fdebye_ext(t,pin[1:comp[0]+2])

    ## einstein
    n = comp[0]+2
    for i in range(comp[1]):
        f += pin[n] * feinstein(t,pin[n+1])

    return f

def scombine(t,pin,comp):
    if np.asarray(t).ndim == 0:
        s = 0
    else:
        s = np.zeros(t.size)

    ## extended debye
    s += pin[0] * sdebye_ext(t,pin[1:comp[0]+2])

    ## einstein
    n = comp[0]+2
    for i in range(comp[1]):
        s += pin[n] * seinstein(t,pin[n+1])

    return s

def cvcombine(t,pin,comp):
    if np.asarray(t).ndim == 0:
        cv = 0
    else:
        cv = np.zeros(t.size)

    ## extended debye
    cv += pin[0] * cvdebye_ext(t,pin[1:comp[0]+2])

    ## einstein
    n = comp[0]+2
    for i in range(comp[1]):
        cv += pin[n] * cveinstein(t,pin[n+1])

    return cv

## read the svib file
xx = np.loadtxt(prefix + ".svib",usecols=(0,1,2,3,4))
f0 = xx[0,1] * z / 4.184 / 627.50947 ## zero-point energy in Ha
t = xx[1:,0] ## temperature in K (skip 0 K)
s = xx[1:,3] * z / 1000 / 4.184 / 627.50947 ## entropy in Ha/K (skip 0 K)

## initial debye fit
def lsqr_residuals_debye(x,*args,**kwargs):
    return (s - sdebye(t,x)) / t

print("--- simple debye model fit ---",flush=True)
res = scipy.optimize.least_squares(lsqr_residuals_debye, 1000,
                                   bounds=(0,np.inf), ftol=1e-12,
                                   xtol=None, gtol=None, verbose=0)
td = res.x[0]
print("Initial debye temperature (K) = %.4f\n" % td,flush=True)

## extended debye fit
def lsqr_residuals_combine(x,*args,**kwargs):
    return s - scombine(t,x,*args)

print("--- combined debye model fit ---",flush=True)
pattern = [npoly_debye,neinstein]
pin = [1.0, td] + npoly_debye * [0.0]
for i in range(neinstein):
    pin = pin + neinstein * [0., (i+1) * 100.]
res = scipy.optimize.least_squares(lsqr_residuals_combine, pin,
                                   method='lm', ftol=1e-15, xtol=1e-15,
                                   gtol=1e-15, verbose=0,args=[pattern])
if not res.success:
    raise Exception("Error in final least squares, cannot continue")
print("Debye model (multiplier = %.10f)" % res.x[0],flush=True)
print("--> Debye temperature (K) = %.4f" % res.x[1],flush=True)
for i in range(npoly_debye):
    print("--> extended coefficient %d = %.10f" % (i+1,res.x[2+i]))
n = npoly_debye + 2
for i in range(neinstein):
    print("Einstein contribution %d (multiplier = %.10f) at %.10f K" % (i+1,res.x[n],res.x[n+1]))
    n = n + 2
print("Final r2 = %.10f\n" % r2_score(s,scombine(t,res.x,pattern)))

## output the parameters in prefix.xdebye
with open(prefix + ".xdebye","w") as f:
    for i in res.x:
        print(i,file=f)

## output the temperatures in thermal-data
with open(prefix + ".thermal-data","w") as f:
    #conver = scipy.constants.physical_constants['Hartree energy in eV'][0] ## results in eV
    conver = 1 ## results in Hartree
    tlist = np.array(temperatures)
    fd = (f0 + fcombine(tlist,res.x,pattern)) * conver
    sd = np.maximum(scombine(tlist,res.x,pattern) * conver,1e-11)
    cd = np.maximum(cvcombine(tlist,res.x,pattern) * conver,1e-11)

    print("## T(K) F(Ha) S(Ha/K) Cv(Ha/K)",file=f,flush=True)
    for x in zip(tlist,fd,sd,cd):
        print("%.2f %.10f %.10f %.10f" % (x[0], x[1], x[2], x[3]),file=f,flush=True)

## create entropy plot
t = xx[:,0]
s = xx[:,3] * z
plt.plot(t,s,'ok',label='$S_{\\rm QP}$ data')
plt.plot(temperatures,scombine(temperatures,res.x,pattern) * 1000 * 4.184 * 627.50947,
         '-r',label='Extended Debye fit')
plt.xlabel('Temperature (K)')
plt.ylabel('Entropy (J/K/mol)')
plt.legend()
plt.savefig(prefix + '.pdf')


## 3a-fit_debye_model-harmonic.py: fit extended debye model to S(T)
## data from the harmonic approximation, and obtain fit parameters and
## list of (T,F,S,Cv).
##
## Input: prefix.info, prefix.fc2_harmonic
## Output: prefix.xdebye_harmonic, prefix.thermal-data_harmonic, prefix_harmonic.pdf
##
## .xdebye_harmonic contains: F0 in hartree, npoly, neinstein, TD, aD0, aD1,..., c1, TE1, c2, TE2,...
##
import numpy as np

## input block ##
prefix="xxxx" ## prefix for the generated files
temperatures = np.arange(0, 1000, 21) # temperature list
npoly_debye=2 # number of parameters in the polynomial part of extended Debye
aeinstein=[1000] # characteristic temperatures for each of the Einstein terms (leave empty for no Einstein terms)
tdinitial = 300. # if None, use the intial Debye fit temperature; otherwise use this value
#################

import pickle
import scipy
import scipy.constants
import matplotlib.pyplot as plt
import sys
import os
sys.stderr = open(os.devnull, 'w')
from pygsl.testing.sf import debye_3 as D3
sys.stderr.close()
sys.stderr = sys.__stderr__
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
    return -kB * t * D3(x) + 3 * kB * t * np.log(1 - np.exp(-x))

def sdebye(t,thetad):
    x = thetad / t
    return -3 * kB * np.log(1 - np.exp(-x)) + 4 * kB * D3(x)

def cvdebye(t,thetad):
    x = thetad / t
    return 12 * kB * D3(x) - 9 * kB * x / (np.exp(x) - 1)

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
    f[idx] = - kB * t[idx] * D3(1/x[idx]) + 3 * kB * t[idx] * np.log(1 - np.exp(-1/x[idx]))

    term = 0.0
    for i in reversed([0] + [-i[1] / (i[0]+2) for i in enumerate(pin[1:])]):
        term = x * term + i
    f += kB * t * term

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
    s[idx] = -3 * kB * np.log(1 - np.exp(-1/x[idx])) + 4 * kB * D3(1/x[idx])

    term = 0.0
    for i in reversed([0] + list(pin[1:])):
        term = x * term + i
    s += kB * term
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
    cv[idx] = 12 * kB * D3(1/x[idx]) - 9 * kB / x[idx] / (np.exp(1/x[idx]) - 1)
    term = 0.0
    for i in reversed([i[1] * (i[0]+1) for i in enumerate(pin[1:])]):
        term = x * term + i
    cv += kB * x * term

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
## pin = [td, -- npoly_debye --, coef_eins, a_eins, coef_eins, a_eins, ...]

def fcombine(t,pin,comp):
    if np.asarray(t).ndim == 0:
        f = 0
    else:
        f = np.zeros(t.size)

    ## einstein
    sumc = 0.
    n = comp[0]+1
    for i in range(comp[1]):
        f += pin[n] * feinstein(t,pin[n+1])
        sumc += pin[n]
        n+=2

    ## extended debye
    f += (1.-sumc) * fdebye_ext(t,pin[:comp[0]+1])

    return natom * f

def scombine(t,pin,comp):
    if np.asarray(t).ndim == 0:
        s = 0
    else:
        s = np.zeros(t.size)

    ## einstein
    sumc = 0.
    n = comp[0]+1
    for i in range(comp[1]):
        s += pin[n] * seinstein(t,pin[n+1])
        sumc += pin[n]
        n+=2

    ## extended debye
    s += (1.-sumc) * sdebye_ext(t,pin[:comp[0]+1])

    return natom * s

def cvcombine(t,pin,comp):
    if np.asarray(t).ndim == 0:
        cv = 0
    else:
        cv = np.zeros(t.size)

    ## einstein
    sumc = 0.
    n = comp[0]+1
    for i in range(comp[1]):
        cv += pin[n] * cveinstein(t,pin[n+1])
        sumc += pin[n]
        n+=2

    ## extended debye
    cv += (1.-sumc) * cvdebye_ext(t,pin[:comp[0]+1])

    return natom * cv

## temperature list for the harmonic calculation
maxt = np.max(temperatures)
tlisth = np.linspace(0,maxt,301)

## get the harmonic FC2
with open(prefix + ".fc2_harmonic","rb") as f:
    fc2 = pickle.load(f)

## harmonic calculation
phcel.force_constants = fc2
phcel.run_mesh(150.,with_eigenvectors=False)
phcel.run_thermal_properties(temperatures=tlisth)
fvib = phcel.get_thermal_properties_dict()['free_energy']
svib = phcel.get_thermal_properties_dict()['entropy']
cv = phcel.get_thermal_properties_dict()['heat_capacity']

# filter out the nans
mask = ~(np.isnan(cv) | np.isnan(svib) | np.isnan(fvib))
tlisth = tlisth[mask]
fvib = fvib[mask]
svib = svib[mask]
cv = cv[mask]

## skip the first temperature, save the zero-point Fvib
f0 = fvib[0] * z / 4.184 / 627.50947 ## zero-point energy in Ha
t = tlisth[1:] ## temperature in K (skip 0 K)
s = svib[1:] * z / 1000 / 4.184 / 627.50947 ## entropy in Ha/K (skip 0 K)

if tdinitial is None:
    ## initial debye fit
    def lsqr_residuals_debye(x,*args,**kwargs):
        return (s - sdebye(t,x)) / t

    print("--- simple debye model fit ---",flush=True)
    res = scipy.optimize.least_squares(lsqr_residuals_debye, 1000,
                                       bounds=(0,np.inf), ftol=1e-12,
                                       xtol=None, gtol=None, verbose=0)
    td = res.x[0]
    print("Initial debye temperature (K) = %.4f\n" % td,flush=True)
else:
    td = tdinitial
    print("Initial debye temperature (K) = %.4f\n" % tdinitial,flush=True)

## extended debye fit
def lsqr_residuals_combine(x,*args,**kwargs):
    return s - scombine(t,x,*args)

print("--- combined debye model fit ---",flush=True)
neinstein = len(aeinstein)
pattern = [npoly_debye,neinstein]
pin = [td] + npoly_debye * [0.0]
for i in range(neinstein):
    pin = pin + [1e-40, aeinstein[i]]

## set bounds
lb = np.zeros((len(pin),)) - np.inf
ub = np.zeros((len(pin),)) + np.inf
lb[0] = 0. ## td
n = npoly_debye+1
for i in range(neinstein):
    lb[n] = 0.
    ub[n] = 1.
    lb[n+1] = 0.
    n+=2

## run the fit
res = scipy.optimize.least_squares(lsqr_residuals_combine, pin, bounds=(lb,ub),
                                   method='trf', ftol=1e-15, xtol=1e-15,
                                   gtol=1e-15, verbose=0,args=[pattern])
if not res.success:
    raise Exception("Error in final least squares, cannot continue")

## calculate the debye coefficient
sumc = 1.
n = npoly_debye+1
for i in range(neinstein):
    sumc -= res.x[n]
    n+=2

print("Debye model (multiplier = %.10f)" % sumc,flush=True)
print("--> Debye temperature (K) = %.4f" % res.x[0],flush=True)
for i in range(npoly_debye):
    print("--> extended coefficient %d = %.10f" % (i+1,res.x[1+i]))
n = npoly_debye + 1
for i in range(neinstein):
    print("Einstein contribution %d (multiplier = %.10f) at %.10f K" % (i+1,res.x[n],res.x[n+1]))
    n = n + 2
print("Final r2 = %.10f\n" % r2_score(s,scombine(t,res.x,pattern)))

## output the parameters in prefix.xdebye_harmonic
## pin = [td, -- npoly_debye --, coef_eins, a_eins, coef_eins, a_eins, ...]
with open(prefix + ".xdebye_harmonic","w") as f:
    print(f0,res.x[0],end=" ",file=f)
    for x_ in res.x[1:1+npoly_debye]:
        print(x_,end=" ",file=f)
    for x_ in res.x[npoly_debye+1::2]:
        print(x_,end=" ",file=f)
    for x_ in res.x[npoly_debye+2::2]:
        print(x_,end=" ",file=f)
    print("",file=f)

## output the temperatures in thermal-data_harmonic
with open(prefix + ".thermal-data_harmonic","w") as f:
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
plt.plot(tlisth,svib * z,'ok',label='$S_{\\rm harm}$ data')
plt.plot(temperatures,scombine(temperatures,res.x,pattern) * 1000 * 4.184 * 627.50947,
         '-r',label='Extended Debye fit')
plt.xlabel('Temperature (K)')
plt.ylabel('Entropy (J/K/mol)')
plt.legend()
plt.savefig(prefix + '_harmonic.pdf')


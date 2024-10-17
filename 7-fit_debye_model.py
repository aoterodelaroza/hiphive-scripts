## 6-fit_debye_model.py: fit extended debye model to S(T) data and obtain
## fit parameters and list of (T,F,S,Cv)
##
## Input: prefix.info, prefix.svib
## Output: prefix.xdebye, prefix.thermal-data, prefix.png

import numpy as np

## input block ##
prefix="urea" ## prefix for the generated files
temperatures = np.arange(0, 3010, 10) # extended temperature list
npoly=3 # number of parameters in the polynomial part of extended Debye
#################

import pickle
import scipy
import scipy.constants
import matplotlib.pyplot as plt
from pygsl.testing.sf import debye_3 as D3

## deactivate deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load the info file
with open(prefix + ".info","rb") as f:
    calculator, maximum_cutoff, acoustic_sum_rules, nthread_batch_lsqr, phcalc, ncell, cell, cell_for_cs, scel, fc_factor, phcel, out_kwargs = pickle.load(f)
z = len(cell.get_chemical_symbols()) / len(phcel.primitive.masses)
kB = scipy.constants.k / scipy.constants.physical_constants['hartree-joule relationship'][0] # Ha/K
natom = len(cell)

# debye and extended debye functions
def fdebye(t,thetad):
    x = thetad / t
    print(natom,kB * t * np.log(1 - np.exp(-x)))
    return -natom * kB * t * D3(x) + 3 * natom * kB * t * np.log(1 - np.exp(-x))

def sdebye(t,thetad):
  x = thetad / t
  return -3 * natom * kB * np.log(1 - np.exp(-x)) + 4 * natom * kB * D3(x)

def cvdebye(t,thetad):
  x = thetad / t
  return 12 * natom * kB * D3(x) - 9 * natom * kB * x / (np.exp(x) - 1)

def fdebye_ext(t,pin):
  thetad = pin[0]

  t = np.asarray(t)
  scalar_input = False
  if t.ndim == 0:
      t = t[None]
      scalar_input = True

  x = t / thetad
  f = np.zeros(x.size) + f0
  idx = t > 1e-5
  f[idx] = f0 - natom * kB * t[idx] * D3(1/x[idx]) + 3 * natom * kB * t[idx] * np.log(1 - np.exp(-1/x[idx]))

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

## read the svib file
xx = np.loadtxt(prefix + ".svib",usecols=(0,1,2,3,4))
f0 = xx[0,1] * z / 4.184 / 627.50947 ## zero-point energy in Ha
t = xx[1:,0] ## temperature in K (skip 0 K)
s = xx[1:,3] * z / 1000 / 4.184 / 627.50947 ## entropy in Ha/K (skip 0 K)

## initial debye fit
def lsqr_residuals_debye(x,*args,**kwargs):
    return (s - sdebye(t,x)) / t

print("--- simple debye model fit ---")
res = scipy.optimize.least_squares(lsqr_residuals_debye, 1000,
                                   bounds=(0,np.inf), ftol=1e-12,
                                   xtol=None, gtol=None, verbose=2)

td = res.x[0]
print("Initial debye temperature (K) = %.4f\n" % td)

## extended debye fit
def lsqr_residuals_extdebye(x,*args,**kwargs):
    return s - sdebye_ext(t,x)

print("--- extended debye model fit ---")
res = scipy.optimize.least_squares(lsqr_residuals_extdebye, [td] + npoly*[0],
                                   method='lm', ftol=1e-15, xtol=1e-15,
                                   gtol=1e-15, verbose=2)

## output the parameters in prefix.xdebye
with open(prefix + ".xdebye","w") as f:
    for i in res.x:
        print(i,file=f)

## output the temperatures in thermal-data
with open(prefix + ".thermal-data","w") as f:
    #conver = scipy.constants.physical_constants['Hartree energy in eV'][0] ## results in eV
    conver = 1 ## results in Hartree
    tlist = np.array(temperatures)
    fd = fdebye_ext(tlist,res.x) * conver
    sd = np.maximum(sdebye_ext(tlist,res.x) * conver,1e-11)
    cd = np.maximum(cvdebye_ext(tlist,res.x) * conver,1e-11)

    print("## T(K) F(Ha) S(Ha/K) Cv(Ha/K)",file=f)
    for x in zip(tlist,fd,sd,cd):
        print("%.2f %.10f %.10f %.10f" % (x[0], x[1], x[2], x[3]),file=f)

## create entropy plot
t = xx[:,0]
s = xx[:,3] * z
plt.plot(t,s,'ok',label='$S_{\\rm QP}$ data')
plt.plot(temperatures,sdebye_ext(temperatures,res.x) * 1000 * 4.184 * 627.50947,
        '-r',label='Extended Debye fit')
plt.xlabel('Temperature (K)')
plt.ylabel('Entropy (J/K/mol)')
plt.legend()
plt.savefig(prefix + '.png')

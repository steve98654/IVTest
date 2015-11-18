from scipy.stats import norm
from scipy.special import ndtr 
from scipy.optimize import fsolve, root 
from math import exp, log, sqrt, pi
from numba import jit 
import hello_ext
import timeit

# TODO:  Do Julia, Try implied vol
# calculator (what optimizer to choose??)  
# https://mpra.ub.uni-muenchen.de/6867/1/MPRA_paper_6867.pdf

# Boost implementation 
# http://www.boost.org/doc/libs/1_42_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/dists/normal_dist.html

## Think of different implied vol approximation formulas 
# http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1965977

## Julia would beat all! 

## Numba is focused on array computing http://numba.pydata.org/numba-doc/0.21.0/user/overview.html
# If we are looping over things in Python, may see performance increase in numba 

# Point 1.7.2.4 of the FAQ indicates that composing @jit functions is preferred 
# http://numba.pydata.org/numba-doc/0.21.0/user/faq.html

# Example where numba really shines, 1.8.2 of http://numba.pydata.org/numba-doc/0.21.0/user/examples.html

# Try numba multi-threading??? 
# I am getting same speed when specifying types of numba function as when just using @jit 

## time it 10000 loops, best of 3: 122 micro secs per loop (no jit)
## 128 micro sec with jit (actually slightly slower with numba)
## getting 3.72 micro sec with ndtr (40x speed up just like owen) (with @jit slower 4.85) 
## Owen call 714 nanosecs 
## Owen with boost norm cdf 923 ns per loop 

@jit
def bs_call(S, K, r, sig, tau):
    stau = sig*sqrt(tau)
    d1 = (log(S/K)+(r+sig*sig/2)*tau)/stau
    d2 = d1 - stau
    return ndtr(d1)*S - ndtr(d2)*K*exp(-r*tau)
    #return norm.cdf(d1)*S - norm.cdf(d2)*K*exp(-r*tau)

## time it 10000 loops, best of 3: 128 micro secs per loop 
@jit('float64(float64,float64,float64,float64,float64,int64)')
def owen_opt_price_test(F, K, vol, t, df, opt_type):
    vol_root_t = vol * sqrt(t)
    d1 = (log(F/K)+0.5*vol_root_t*vol_root_t)/vol_root_t
    d2=d1-vol_root_t
    px=opt_type*F*norm.cdf(opt_type*d1)-opt_type*K*norm.cdf(opt_type*d2)
    return px*df

## Corrado and Miller Initial Implied Vol 
def cm_initial_iv(C,S,K,tau):
    sk = S-K
    csk = (C-0.5*(S-K))*(C-0.5*(S-K))
    return sqrt(2*pi/tau)/(S+K)*(C-0.5*(S-K)+sqrt(csk*csk-sk*sk/pi))

## 203 micro secs per loop
def iv_finder(C, S, K, r, tau):
    return fsolve(lambda x: C-bs_call(S,K,r,x,tau), cm_initial_iv(C,S,K,tau))

## getting 55.4 micro secs per loop 
def iv_finder2(C, S, K, r, tau):
    return fsolve(lambda x: C-hello_ext.BlackScholesCall(S,K,r,float(x),tau), cm_initial_iv(C,S,K,tau))

## root tests 
## hybr 52.6
## lm 140 
## broyden1 967
## broyden2 905
## anderson 897
## linearmixing 1601
## diagbroyden 664
## excitingmixing 1260
## krylov 1401
## df-sane 233
def iv_finder3(C, S, K, r, tau):
    return root(lambda x: C-hello_ext.BlackScholesCall(S,K,r,float(x),tau),\
            cm_initial_iv(C,S,K,tau), method='hybr')

## http://jaeckel.org/  (Paper called By Implication)

## Bring Examples together and make them easy to use 

print hello_ext.greet()
print hello_ext.BlackScholesCall(1.1, 1.0, 0.01, 0.3, 0.5)
print hello_ext.OwenOptPriceTest(1.1, 1.0, 0.03, 0.5, 1.0, 1)
print iv_finder(2.00, 51.25, 50., 0.05, 0.08767)
print bs_call(1.1, 1.0, 0.01, 0.3, 0.5) 
        

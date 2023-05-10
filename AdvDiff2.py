import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from numba import jit

params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params) 

pts_to_inches = 1/72.27

fig_width_inches = pts_to_inches*443.57848
fig_height_inches = fig_width_inches*(-1+np.sqrt(5))/2

##################################################

epsilon= 5e-2

@jit
def b(y):
    return np.exp(-y)

b_bar = 1/(np.e-1)

@jit
def u_0(x,t):
    return np.exp(-(x+b_bar*t)**2)

@jit
def du_0dx(x,t):
    return -2*(x+b_bar*t)*u_0(x,t)

@jit
def chi(y):
    return b_bar*np.exp(y) -y - 0.5

@jit
def u_1(x,t,epsilon):
    y = np.remainder(x/epsilon,1)
    return du_0dx(x,t)*chi(y)

@jit
def RHS_full(t,u,x,epsilon):
    dx = x[1]-x[0]
    dudt = np.zeros_like(x)
    dudx = np.zeros_like(x)
    for i in range(0,len(x)-1):
        dudx[i] = (u[i+1]-u[i])/dx
        dudt[i] = b(np.remainder(x[i]/epsilon,1))*dudx[i]  
    dudt[-1]=0
    return dudt

@jit
def initial_data(x):
    return np.exp(-x**2)

#spacetime discretisation
x = np.linspace(-3,3,3000)
t_max = 1
t_vals = np.linspace(0,t_max,20)

#scipy evolution:
soln_full = scipy.integrate.solve_ivp(RHS_full,[0,t_max],initial_data(x),dense_output=True,args=(x,epsilon,),rtol=1e-6,atol=1e-8)

#manual evolution:
# def ssperk33(f, yn, dt):
#     y1 = yn + dt * f(yn)
#     y2 = (3 * yn + y1 + dt * f(y1)) / 4
#     return (yn + 2 * y2 + 2 * dt * f(y2)) / 3

# def RHS_full_ssperk(x, epsilon):
#     return lambda u : RHS_full(0, u, x, epsilon)

# def doEvoution(x,epsilon):
#     dt_orig = 0.1/b_bar * (x[1] - x[0])**2
#     dt = dt_orig
#     t_current = 0
#     u_full = initial_data(x)
#     next_t_i = 1
#     u_full_all = np.zeros((len(x), len(t_vals)))
#     u_full_all[:, 0] = u_full

#     while t_current < t_max:
#         if t_current + dt > t_vals[next_t_i]:
#             dt = t_vals[next_t_i] - t_current
#             print(t_current)
#             next_t_i += 1
#             flag = True
#         else:
#             dt = dt_orig
#             flag = False
#         t_current += dt
#         u_full = ssperk33(RHS_full_ssperk(x, epsilon), u_full, dt)    
#         if flag:
#             u_full_all[:, next_t_i-1] = u_full
#     return u_full_all

#soln_full_manual = doEvoution(x,epsilon)


fig,ax = plt.subplots(1,3,figsize=(10,5))
plt.subplots_adjust(top=0.89, right=0.77)
plt.figure(1)
for t in t_vals:
    ax[0].plot(x,soln_full.sol(t))
    ax[1].plot(x,u_0(x,t))
    ax[2].plot(x,u_0(x,t)+epsilon*u_1(x,t,epsilon),label=rf'$t= {np.round(t,2)}$')
for i in [0,1,2]:
    ax[i].grid()    
    ax[i].set_xlabel(r'$x$')
ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[0].set_ylabel(r'$u$')
ax[1].set_ylabel(r'$u_0$')
ax[2].set_ylabel(r'$u_0 + \varepsilon u_1$')
ax[0].set_title('Full Solution')
ax[1].set_title('Leading order average solution')
ax[2].set_title('First order average solution')
plt.suptitle(rf'$\varepsilon = {epsilon}$')
ax[0].set_position([0.06,0.11,0.22,0.78])
ax[1].set_position([0.35,0.11,0.22,0.78])
ax[2].set_position([0.64,0.11,0.22,0.78])
plt.savefig("adv-diff2fig1.pdf", format="pdf", bbox_inches="tight")

fig,ax = plt.subplots(2,2,figsize=(8,5))
plt.figure(2)
ax[0,0].plot(x,soln_full.sol(t_max),'-k',label=rf'$u$')
ax[0,0].plot(x,u_0(x,t_max),'-.r',label=rf'$u_0$')
ax[0,0].grid()
ax[0,0].legend()
ax[0,0].set_xlabel(r'$x$')
ax[0,0].set_ylabel(r'$u$')
ax[0,0].set_title('Full and leading order solutions')
ax[0,0].set_position([0.105,0.575,0.375,0.33])

ax[0,1].plot(x,soln_full.sol(t_max),'-k',label=rf'$u$')
ax[0,1].plot(x,u_0(x,t_vals[-1])+epsilon*u_1(x,t_vals[-1],epsilon),'--b',label=rf'$u_0 + \varepsilon u_1$')
ax[0,1].grid()
ax[0,1].legend()
ax[0,1].set_xlabel(r'$x$')
ax[0,1].set_ylabel(r'$u$')
ax[0,1].set_title('Full and first order solutions')
ax[0,1].set_position([0.6,0.575,0.375,0.33])

ax[1,0].plot(x,soln_full.sol(t_max)- u_0(x,t_max) ,'-.r')
ax[1,0].grid()
ax[1,0].set_xlabel(r'$x$')
ax[1,0].set_ylabel(r'$u - u_0$')
ax[1,0].set_title('Error in leading order solution')
ax[1,0].set_position([0.105,0.1,0.375,0.33])

ax[1,1].plot(x,soln_full.sol(t_max)- u_0(x,t_max)- epsilon*u_1(x,t_max,epsilon),'--b')
ax[1,1].grid()
ax[1,1].set_xlabel(r'$x$')
ax[1,1].set_ylabel(r'$u - (u_0+ \varepsilon u_1)$')
ax[1,1].set_title('Error in first order solution')
ax[1,1].set_position([0.6,0.1,0.375,0.33])

plt.suptitle(rf'$t={t_max}$, $\varepsilon = {epsilon}$')
plt.savefig("adv-diff2fig2.pdf", format="pdf", bbox_inches="tight")

epsilons = 1/2**np.arange(3,9)
errors1 = np.zeros_like(epsilons)
errors2 = np.zeros_like(epsilons)

for i, epsilon in enumerate(epsilons):
    x,dx=np.linspace(-3,3,40000,retstep=True)
    soln_full = scipy.integrate.solve_ivp(RHS_full,[0,t_max],initial_data(x),dense_output=True,args=(x,epsilon,),rtol=1e-8,atol=1e-10)
    #soln_full_manual = doEvoution(x,epsilon)
    errors1[i] = np.linalg.norm(abs(soln_full.sol(t_max) - u_0(x,t_max)),1)*dx
    errors2[i] = np.linalg.norm(abs(soln_full.sol(t_max) - u_0(x,t_max)-epsilon*u_1(x,t_max,epsilon)),1)*dx

errors1_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors1),1)
errors2_polyfit_coeffs = np.polyfit(np.log(epsilons[:4]),np.log(errors2[:4]),1)

def trendline(data,x):
    return np.poly1d(data)(x)

plt.figure(3,figsize=(fig_width_inches,fig_height_inches))
plt.loglog(epsilons,np.exp(trendline(errors1_polyfit_coeffs,np.log(epsilons))),'-r',label=rf'$\propto \varepsilon^{{{np.around(errors1_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons[:4],np.exp(trendline(errors2_polyfit_coeffs,np.log(epsilons)))[:4],'--',c='orange',label=rf'$\propto \varepsilon^{{{np.around(errors2_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons,errors1,'xk',label='Error at leading order')
plt.loglog(epsilons,errors2,'.b',label='Error at first order')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$\left\Vert\mathrm{Error}\right\Vert_{1}$')
plt.title(r'Error against $\varepsilon$')
plt.legend()
plt.tight_layout()
plt.savefig("adv-diff2fig3.pdf", format="pdf", bbox_inches="tight")


plt.show()
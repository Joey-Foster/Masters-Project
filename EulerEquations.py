import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params) 

pts_to_inches = 1/72.27

fig_width_inches = pts_to_inches*443.57848
fig_height_inches = fig_width_inches*(-1+np.sqrt(5))/2

##################################################

theta = 1
epsilon = 1e-2

def p(rho, e ,Y):
    return Y*rho*e
        
def Gamma(rho,e,Y):
    return rho/3 - e*Y

def Y_eq(rho,e):
    return rho/(3*e)

def RHS_full(t,z,epsilon):
    rho , e , Y = z
    return [-rho*theta, -theta/rho * p(rho,e,Y), 1/epsilon * Gamma(rho,e,Y) ]

def RHS_average(t,x, epsilon):
    rho, e = x
    return [-rho*theta, -theta/rho * p(rho,e,Y_eq(rho,e)) + epsilon*correction(rho,e)]

def correction(rho,e):
    return -theta**2 * (rho/(3*e)*(1-rho/(3*e)))
         
soln_full = scipy.integrate.solve_ivp(RHS_full,[0,1],[1,1,1],dense_output=True, args=(epsilon,)) #Y(0)=1 so fast decay is visible on this example plot

t = np.linspace(0,1,101)
labels = [r'$\rho$',r'$e$',r'$Y$']
fig, axes = plt.subplots(1,3,figsize=(fig_width_inches,fig_height_inches))
plt.figure(1)
for i, ax in enumerate(axes):
    ax.plot(t,soln_full.sol(t)[i,:])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(labels[i])
    ax.grid()
    ax.set_xlim(0,1)
plt.tight_layout()
fig.subplots_adjust(top=0.9)
plt.suptitle('Full Solution')
plt.savefig("figure1.pdf", format="pdf", bbox_inches="tight")

soln_full = scipy.integrate.solve_ivp(RHS_full,[0,1],[1,1,1/3],dense_output=True, args=(epsilon,)) #Y(0)=Y_eq(0)

soln_leading_order = scipy.integrate.solve_ivp(RHS_average,[0,1],[1,1],dense_output=True,rtol=1e-6,atol=1e-8, args=(0,))

soln_first_order = scipy.integrate.solve_ivp(RHS_average,[0,1],[1,1],dense_output=True, args=(epsilon,))

fig, axes = plt.subplots(1,2,figsize=(fig_width_inches,fig_height_inches))
plt.figure(2)
axes[0].plot(t,soln_full.sol(t)[1,:],c='black',label=r'$e$')
axes[0].plot(t,soln_leading_order.sol(t)[1,:],'-.',c='red',label=r'$\bar{e}_{\mathrm{leading \; order}}$')
axes[0].plot(t,soln_first_order.sol(t)[1,:],'--',c='blue',label=r'$\bar{e}_{\mathrm{first \; order}}$')
axes[0].set_xlabel(r'$t$')
axes[0].set_ylabel(r'$e$')
axes[0].set_xlim(0,1)
axes[0].grid()
axes[0].set_title('Full solution and both averaged \n' rf'solutions for $e(t)$ at $\varepsilon = {epsilon}$')
axes[0].legend(fontsize=8)

leading_difference = soln_full.sol(t)[1,:] - soln_leading_order.sol(t)[1,:]
first_order_difference = soln_full.sol(t)[1,:] - soln_first_order.sol(t)[1,:]

axes[1].plot(t,leading_difference,'-.',c='red',label=r'$e - \bar{e}_{\mathrm{leading \; order}}$')
axes[1].plot(t,first_order_difference,'--',c='blue',label=r'$e - \bar{e}_{\mathrm{first \; order}}$')
axes[1].set_xlabel(r'$t$')
axes[1].set_ylabel(r'$e - \bar{e}$')
axes[1].set_xlim(0,1)
axes[1].grid()
axes[1].set_title('Error in both averaged \n' rf'solutions for $e(t)$ at $\varepsilon = {epsilon}$')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("figure2.pdf", format="pdf", bbox_inches="tight")

epsilons = 0.1/2**(np.arange(1,10))
errors = np.zeros_like(epsilons)
errors_first_order = np.zeros_like(epsilons)


for i, epsilon in enumerate(epsilons):

    soln_full = scipy.integrate.solve_ivp(RHS_full,[0,1],[1,1,1/3],dense_output=True,rtol=1e-6,atol=1e-8, args=(epsilon,)) 

    soln_first_order = scipy.integrate.solve_ivp(RHS_average,[0,1],[1,1],dense_output=True,rtol=1e-8,atol=1e-10, args=(epsilon,))

    errors[i] = np.linalg.norm(soln_full.sol(t)[1,:] - soln_leading_order.sol(t)[1,:],1)
    errors_first_order[i] = np.linalg.norm(soln_full.sol(t)[1,:] - soln_first_order.sol(t)[1,:],1)

errors_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors),1)

errors_first_order_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors_first_order),1)

def errors_polyfit(x):
    return np.poly1d(errors_polyfit_coeffs)(x)

def errors_first_order_polyfit(x):
    return np.poly1d(errors_first_order_polyfit_coeffs)(x)

plt.figure(3,figsize=(fig_width_inches,fig_height_inches))
plt.loglog(epsilons,np.exp(errors_polyfit(np.log(epsilons))),'-r',label=rf'$\propto \varepsilon^{{{np.around(errors_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons,np.exp(errors_first_order_polyfit(np.log(epsilons))),'--',c='orange',label=rf'$\propto \varepsilon^{{{np.around(errors_first_order_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons, errors, 'x',lw=2,label='Error at leading order',c='k')
plt.loglog(epsilons,errors_first_order,'.',label='Error at first order',c='b')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$\left\Vert\mathrm{Error}\right\Vert_{1}$')
plt.title(r'Error against $\varepsilon$')
plt.legend()
plt.savefig("figure3.pdf", format="pdf", bbox_inches="tight")



##################################################
#limit cycle
epsilon = 5e-2

def RHS_cycle(t,z,epsillon):
    rho, e, Y1, Y2 = z
    return [-rho*theta,-theta/2 *(Y1+Y2)*e,1/epsilon *(-Y2 + rho/(3*e)), 1/epsilon*(Y1 - rho/(3*e))]

soln_cycle = scipy.integrate.solve_ivp(RHS_cycle,[0,1],[1,1,2/5,2/5],dense_output=True, args=(epsilon,),rtol=1e-8,atol=1e-10)

soln_leading_order = scipy.integrate.solve_ivp(RHS_average,[0,1],[1,1],dense_output=True, args=(0,))

t = np.linspace(0,1,201)
labels = [r'$\rho$',r'$e$',r'$Y_1$',r'$Y_2$']
fig, axes = plt.subplots(2,2,figsize=(fig_width_inches,fig_height_inches))
plt.figure(4,figsize=(2,1.5))
axes[0,0].plot(t,soln_cycle.sol(t)[0,:])
axes[0,0].set_xlabel(r'$t$')
axes[0,0].set_ylabel(labels[0])
axes[0,0].grid()
axes[0,0].set_xlim(0,1)

axes[0,1].plot(t,soln_cycle.sol(t)[1,:])
axes[0,1].set_xlabel(r'$t$')
axes[0,1].set_ylabel(labels[1])
axes[0,1].grid()
axes[0,1].set_xlim(0,1)

axes[1,0].plot(t,soln_cycle.sol(t)[2,:])
axes[1,0].set_xlabel(r'$t$')
axes[1,0].set_ylabel(labels[2])
axes[1,0].grid()
axes[1,0].set_xlim(0,1)

axes[1,1].plot(t,soln_cycle.sol(t)[3,:])
axes[1,1].set_xlabel(r'$t$')
axes[1,1].set_ylabel(labels[3])
axes[1,1].grid()
axes[1,1].set_xlim(0,1)
plt.tight_layout()
fig.subplots_adjust(top=0.9)
plt.suptitle('Limit Cycle Solution')
plt.savefig("figure4.pdf", format="pdf", bbox_inches="tight")


fig, ax = plt.subplots(1,2,figsize=(fig_width_inches,fig_height_inches))
plt.figure(5)
ax[0].plot(t,soln_cycle.sol(t)[1,:],c='black',label=r'$e$')
ax[0].plot(t,soln_leading_order.sol(t)[1,:],'-.',c='red',label=r'$\bar{e}_{\mathrm{leading \; order}}$')
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$e$')
ax[0].set_xlim(0,1)
ax[0].grid()
ax[0].set_title('Full and leading order average \n' rf'solutions for $e(t)$ at $\varepsilon = {epsilon}$')
ax[0].legend(fontsize=10)

difference = soln_cycle.sol(t)[1,:] - soln_leading_order.sol(t)[1,:]

ax[1].plot(t,difference,c='red',label=r'$e - \bar{e}_{\mathrm{leading \; order}}$')
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$e - \bar{e}$')
ax[1].set_xlim(0,1)
ax[1].grid()
ax[1].set_title('Error in leading order \n' rf'solution for $e(t)$ at $\varepsilon = {epsilon}$')
ax[1].legend(fontsize=10)
plt.tight_layout()
plt.savefig("figure5.pdf", format="pdf", bbox_inches="tight")

soln_leading_order = scipy.integrate.solve_ivp(RHS_average,[0,1],[1,1],dense_output=True,rtol=1e-8,atol=1e-10, args=(0,))

epsilons = 0.1/2**(np.arange(1,10))
errors = np.zeros_like(epsilons)

for i, epsilon in enumerate(epsilons):

    soln_cycle = scipy.integrate.solve_ivp(RHS_cycle,[0,1],[1,1,2/5,2/5],dense_output=True,rtol=1e-8,atol=1e-10, args=(epsilon,)) 

    errors[i] = np.linalg.norm(soln_cycle.sol(t)[1,:] - soln_leading_order.sol(t)[1,:],1)

errors_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors),1)

def errors_cycle_polyfit(x):
    return np.poly1d(errors_polyfit_coeffs)(x)

plt.figure(6,figsize=(fig_width_inches,fig_height_inches))
plt.loglog(epsilons,np.exp(errors_cycle_polyfit(np.log(epsilons))),c='r',label=rf'$\propto \varepsilon^{{{np.around(errors_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons, errors, 'x',lw=2,label='Error at leading order',c='k')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$\left\Vert\mathrm{Error}\right\Vert_{1}$')
plt.title(r'Error against $\varepsilon$')
plt.legend()
plt.tight_layout()
plt.savefig("figure6.pdf", format="pdf", bbox_inches="tight")

plt.show()
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


epsilon=5e-2

def A(y):
    return 1 + 1/2*np.sin(2*np.pi*y)

def f(x):
    return 2*np.pi*x*(3*x-2)*np.cos(np.pi*x)-(np.pi**2*x**3-np.pi**2*x**2-6*x+2)*np.sin(np.pi*x)

def u_exact(x):
    return x**2*(1-x)*np.sin(np.pi*x)

def u_0(x):
    return 2/np.sqrt(3)*u_exact(x)

def antideriv_of_1_over_A(y):
    return 2/(np.pi*np.sqrt(3))*np.arctan((1+2*np.tan(np.pi*y))/np.sqrt(3))

y_integral = scipy.integrate.quad(antideriv_of_1_over_A,-0.5,0.5)[0]

c_2 = - np.sqrt(3)/2*y_integral

def chi(y):
    return -y + np.sqrt(3)/2 * antideriv_of_1_over_A(y) + c_2

def u_1(x):
    y = np.remainder(x/epsilon,1)-0.5
    return chi(y)*2/np.sqrt(3)*x*((2-3*x)*np.sin(np.pi*x)+np.pi*(1-x)*x*np.cos(np.pi*x))
              
def rhs_full(x,z):
    u,v = z
    rhs=np.zeros_like(z)
    rhs[0] = v/A(np.remainder(x/epsilon,1)-0.5)
    rhs[1] = -f(x)
    return rhs

def bcs(ua,ub):
    return np.array([ua[0],ub[0]])

x_grid = np.linspace(0,1,5)
u_grid = np.zeros((2,5))

soln_full = scipy.integrate.solve_bvp(rhs_full,bcs,x_grid,u_grid,tol=1e-7)

x=np.linspace(0,1,201)

#figsize=(fig_width_inches,fig_height_inches)
fig, ax = plt.subplots(2,2,figsize=(8,5))
fig.subplots_adjust(top=0.8)
plt.figure(1)
ax[0,0].plot(x,soln_full.sol(x)[0,:],'-k',label=r'$u$')
ax[0,0].plot(x,u_exact(x),'-.',c='b',label=r'$\tilde{u}$')
ax[0,0].set_xlabel(r'$x$')
ax[0,0].set_ylabel(r'$u$')
ax[0,0].grid()
ax[0,0].set_title('Full and naive solutions')
ax[0,0].legend()

ax[0,1].plot(x,soln_full.sol(x)[0,:],'-k',label=r'$u$')
ax[0,1].plot(x,u_0(x),'-.r',label=r'$u_0$')
ax[0,1].plot(x,u_0(x) + epsilon*u_1(x),'--b',label=r'$u_0 + \varepsilon u_1$')
ax[0,1].set_xlabel(r'$x$')
ax[0,1].set_ylabel(r'$u$')
ax[0,1].grid()
ax[0,1].set_title('Full and average solutions')
ax[0,1].legend()

ax[1,0].plot(x,soln_full.sol(x)[0,:] - u_0(x),'-.r',label=r'$u - u_0$')
ax[1,0].set_xlabel(r'$x$')
ax[1,0].set_ylabel(r'$u - u_0$')
ax[1,0].grid()
ax[1,0].set_title('Error in leading order solution')

ax[1,1].plot(x,soln_full.sol(x)[0,:] - u_0(x) - epsilon*u_1(x),'--b',label=r'$u - (u_0 + \varepsilon u_1)$')
ax[1,1].set_xlabel(r'$x$')
ax[1,1].set_ylabel(r'$u - (u_0+ \varepsilon u_1)$')
ax[1,1].grid()
ax[1,1].set_title('Error in first order solution')
plt.tight_layout()
plt.suptitle(rf'$\varepsilon = {epsilon}$',y=1)
plt.savefig("elliptic_figure1.pdf", format="pdf", bbox_inches="tight")

epsilons = 1/2**np.arange(2,10)
errors1 = np.zeros_like(epsilons)
errors2 = np.zeros_like(epsilons)

for i, epsilon in enumerate(epsilons):
    x,dx=np.linspace(0,1,1001,retstep=True)
    soln_full = scipy.integrate.solve_bvp(rhs_full,bcs,x_grid,u_grid,tol=1e-6,max_nodes=1e6)
    errors1[i] = np.linalg.norm(abs(soln_full.sol(x)[0,:] - u_0(x)),1)*dx
    errors2[i] = np.linalg.norm(abs(soln_full.sol(x)[0,:] - u_0(x)-epsilon*u_1(x)),1)*dx

errors1_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors1),1)
errors2_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors2),1)

def trendline(data,x):
    return np.poly1d(data)(x)

plt.figure(2,figsize=(fig_width_inches,fig_height_inches))
plt.loglog(epsilons,np.exp(trendline(errors1_polyfit_coeffs,np.log(epsilons))),'-r',label=rf'$\propto \varepsilon^{{{np.around(errors1_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons,np.exp(trendline(errors2_polyfit_coeffs,np.log(epsilons))),'--',c='orange',label=rf'$\propto \varepsilon^{{{np.around(errors2_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons,errors1,'xk',label='Error at leading order')
plt.loglog(epsilons,errors2,'.b',label='Error at first order')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$\left\Vert\mathrm{Error}\right\Vert_{1}$')
plt.title(r'Error against $\varepsilon$')
plt.legend()
plt.savefig("elliptic_figure2.pdf", format="pdf", bbox_inches="tight")


####################################
#boundary layers

epsilon = 5e-2

def bl_f(x):
    return (np.pi**2*x**2 - 2)*np.sin(np.pi*x)-4*np.pi*x*np.cos(np.pi*x)

def bl_u_0(x):
    return 2/np.sqrt(3)*x**2*np.sin(np.pi*x)

def bl_u_1(x):
    y = np.remainder(x/epsilon,1)-0.5
    return chi(y)*(4*x/np.sqrt(3)*np.sin(np.pi*x)+ 2*np.pi*x**2/np.sqrt(3)*np.cos(np.pi*x))

def bl_rhs_full(x,z):
    u,v = z
    rhs=np.zeros_like(z)
    rhs[0] = v/A(np.remainder(x/epsilon,1)-0.5)
    rhs[1] = -bl_f(x)
    return rhs

bl_soln_full = scipy.integrate.solve_bvp(bl_rhs_full,bcs,x_grid,u_grid,tol=1e-7)

def q(x):
    return epsilon*(-bl_u_1(1)*x+bl_u_1(0)*(x-1))

fig, ax = plt.subplots(2,3,figsize=(10,5))
fig.subplots_adjust(top=0.8)
plt.figure(3)
ax[0,0].plot(x,bl_soln_full.sol(x)[0,:],'-k',label=r'$u$')
ax[0,0].plot(x,bl_u_0(x),'-.r',label=r'$u_0$')
ax[0,0].set_xlabel(r'$x$')
ax[0,0].set_ylabel(r'$u$')
ax[0,0].grid()
ax[0,0].set_title('Full and leading order solutions')
ax[0,0].legend()

ax[0,1].plot(x,bl_soln_full.sol(x)[0,:],'-k',label=r'$u$')
ax[0,1].plot(x,bl_u_0(x) + epsilon*bl_u_1(x),'--b',label=r'$u_0 + \varepsilon u_1$')
ax[0,1].set_xlabel(r'$x$')
ax[0,1].set_ylabel(r'$u$')
ax[0,1].grid()
ax[0,1].set_title('Full and first order solutions \n' 'without the boundary correction')
ax[0,1].legend()

ax[0,2].plot(x,bl_soln_full.sol(x)[0,:],'-k',label=r'$u$')
ax[0,2].plot(x,bl_u_0(x) + epsilon*bl_u_1(x)+q(x),':',c='#fc8c03',lw='2',label=r'$u_0 + \varepsilon u_1+q$')
ax[0,2].set_xlabel(r'$x$')
ax[0,2].set_ylabel(r'$u$')
ax[0,2].grid()
ax[0,2].set_title('Full and first order solutions \n' 'with the boundary correction')
ax[0,2].legend(loc='upper left')

ax[1,0].plot(x,bl_soln_full.sol(x)[0,:] - bl_u_0(x),'-.r',label=r'$u - u_0$')
ax[1,0].set_xlabel(r'$x$')
ax[1,0].set_ylabel(r'$u - u_0$')
ax[1,0].grid()
ax[1,0].set_title('Error in leading order solution')

ax[1,1].plot(x,bl_soln_full.sol(x)[0,:] - bl_u_0(x) - epsilon*bl_u_1(x),'--b',label=r'$u - (u_0 + \varepsilon u_1)$')
ax[1,1].set_xlabel(r'$x$')
ax[1,1].set_ylabel(r'$u - (u_0+ \varepsilon u_1)$')
ax[1,1].grid()
ax[1,1].set_title('Error in first order solution \n' 'without the boundary correction')

ax[1,2].plot(x,bl_soln_full.sol(x)[0,:] - bl_u_0(x) - epsilon*bl_u_1(x)-q(x),':',c='#fc8c03',lw='2',label=r'$u - (u_0 + \varepsilon u_1+q)$')
ax[1,2].set_xlabel(r'$x$')
ax[1,2].set_ylabel(r'$u - (u_0+ \varepsilon u_1+q)$')
ax[1,2].grid()
ax[1,2].set_title('Error in first order solution \n' 'with the boundary correction')

plt.tight_layout()
plt.suptitle(rf'$\varepsilon = {epsilon}$',y=1)
plt.savefig("elliptic_figure3.pdf", format="pdf", bbox_inches="tight")


epsilons = 0.1/2**np.arange(1,10)
errors1 = np.zeros_like(epsilons)
errors2 = np.zeros_like(epsilons)
errors3 = np.zeros_like(epsilons)

for i, epsilon in enumerate(epsilons):
    x,dx=np.linspace(0,1,1001,retstep=True)
    bl_soln_full = scipy.integrate.solve_bvp(bl_rhs_full,bcs,x_grid,u_grid,tol=1e-6,max_nodes=1e6)
    errors1[i] = np.linalg.norm(abs(bl_soln_full.sol(x)[0,:] - bl_u_0(x)),1)*dx
    errors2[i] = np.linalg.norm(abs(bl_soln_full.sol(x)[0,:] - bl_u_0(x)-epsilon*bl_u_1(x)),1)*dx
    errors3[i] = np.linalg.norm(abs(bl_soln_full.sol(x)[0,:] - bl_u_0(x)-epsilon*bl_u_1(x)-q(x)),1)*dx

errors1_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors1),1)
errors2_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors2),1)
errors3_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors3),1)

plt.figure(5,figsize=(8,5))
plt.loglog(epsilons,np.exp(trendline(errors1_polyfit_coeffs,np.log(epsilons))),'-r',label=rf'$\propto \varepsilon^{{{np.around(errors1_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons,np.exp(trendline(errors2_polyfit_coeffs,np.log(epsilons))),'--',c='orange',label=rf'$\propto \varepsilon^{{{np.around(errors2_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons,np.exp(trendline(errors3_polyfit_coeffs,np.log(epsilons))),'-.',c='green',label=rf'$\propto \varepsilon^{{{np.around(errors3_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons,errors1,'xk',label='Error at leading order')
plt.loglog(epsilons,errors2,'.b',label='Error at first order \n' 'without boundary correction')
plt.loglog(epsilons,errors3,'*',c='#f531f5',label='Error at first order \n' 'with boundary correction')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$\left\Vert\mathrm{Error}\right\Vert_{1}$')
plt.title(r'Error against $\varepsilon$')
plt.legend()
plt.savefig("elliptic_figure4.pdf", format="pdf", bbox_inches="tight")
plt.show()
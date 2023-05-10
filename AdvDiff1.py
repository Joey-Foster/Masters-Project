import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import sympy as sp
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

yy,dd = sp.symbols('y D')

labels = [r'$V(y)=\frac{1}{5}\sin(2\pi y)$',r'$V(y)=\frac{1}{2}\sin(2\pi y)$',r'$V(y)=\sin(2\pi y)$']
line_styles = ['-k','--r','-.b']
vvs = np.array([0.2*sp.sin(2*sp.pi*yy),0.5*sp.sin(2*sp.pi*yy),sp.sin(2*sp.pi*yy)])
bbs=np.zeros_like(vvs)
rhorhos=np.zeros_like(vvs)

for i,vv in enumerate(vvs):
    bbs[i] = -sp.diff(vv,yy)
    rhorhos[i] = sp.exp(-vv/dd)

class functions:

    def __init__(self,expression):
        self.expression = expression
        self.return_function = ""

    def lambdify(self):
        self.lambdified = sp.lambdify([yy,dd],self.expression,'numpy')
    
    def evaluate_at_D(self):
        def oneDgen(y):
            return self.lambdified(y,D)
        self.return_function = oneDgen


rhos = np.zeros_like(rhorhos)
one_over_rhos = np.zeros_like(rhorhos)

for i,rhorho in enumerate(rhorhos):
    rho = functions(rhorho)
    rho.lambdify()
    rho.evaluate_at_D()
    rhos[i] = rho.return_function

    one_over_rho = functions(1/rhorho)
    one_over_rho.lambdify()
    one_over_rho.evaluate_at_D()
    one_over_rhos[i] = one_over_rho.return_function

Ds = 10**np.linspace(-2,0,200)
n_kappas = np.array([np.zeros_like(Ds)]*len(vvs))


fig, ax = plt.subplots(1,2,figsize=(fig_width_inches,fig_height_inches))
fig.subplots_adjust(top=0.9)
plt.figure(1)

for i in range(len(n_kappas)):
    for j,D in enumerate(Ds):
        z = scipy.integrate.quad(rhos[i],0,1)[0]
        z_bar = scipy.integrate.quad(one_over_rhos[i],0,1)[0]
        n_kappas[i,j]=D/(z*z_bar)

    ax[0].loglog(Ds,n_kappas[i,:],line_styles[i],label=labels[i])
ax[0].set_xlabel(r'$D$')
ax[0].set_ylabel(r'$\mathcal{K}$')


Ds = 10**np.linspace(0,2,200)
for i in range(len(n_kappas)):
    for j,D in enumerate(Ds):
        z = scipy.integrate.quad(rhos[i],0,1)[0]
        z_bar = scipy.integrate.quad(one_over_rhos[i],0,1)[0]
        n_kappas[i,j]=D/(z*z_bar)

    ax[1].loglog(Ds,n_kappas[i,:],line_styles[i])
ax[1].set_xlabel(r'$D$')
ax[0].legend()
fig.suptitle('Effective Diffusion against Molecular Diffusion',y=0.99)
plt.savefig("adv-diff1fig1.pdf", format="pdf", bbox_inches="tight")

################################################

epsilon = 5e-2
D = 1e-1

#calculations for specific choice of b
@jit
def b_func(x,epsilon):
    y = np.remainder(x/epsilon,1)
    return -2*np.pi*0.1*np.cos(2*np.pi*y) #explictly written out -dV/dy because numba doesn't like working with global sympy for some reason

vv = 0.1*sp.sin(2*sp.pi*yy)
rhorho = sp.exp(-vv/dd)
rho = functions(rhorho)
rho.lambdify()
rho.evaluate_at_D()
rho = rho.return_function

one_over_rho = functions(1/rhorho)
one_over_rho.lambdify()
one_over_rho.evaluate_at_D()
one_over_rho = one_over_rho.return_function

z = scipy.integrate.quad(rho,0,1)[0]
z_bar = scipy.integrate.quad(one_over_rho,0,1)[0]

Kappa = D/(z*z_bar)


@jit
def RHS_full(t,u,x,epsilon):
    dx = x[1]-x[0]
    dudt = np.zeros_like(x)
    dudx = np.zeros_like(x)
    for i in range(1,len(x)-1):
        # if b_func(x[i]/epsilon) >0:
        #     dudx[i] = (u[i+1] - u[i])/dx
        # else:
        #     dudx[i] = (u[i] - u[i-1])/dx
        dudx[i] = (u[i+1]-u[i-1])/(2*dx)
        dudt[i] = b_func(x[i],epsilon)/epsilon*dudx[i] + D*(u[i+1]+u[i-1]-2*u[i])/dx**2
    u[0]=0
    u[-1]=0
    return dudt

@jit
def u_naive(x,t):
    return np.sin(np.pi*x)*np.exp(-D*np.pi**2*t)

@jit
def u_0(x,t):
    return np.sin(np.pi*x)*np.exp(-Kappa*np.pi**2*t)

# @jit
# def RHS_no_advection(t,u,x):
#     dx = x[1]-x[0]
#     dudt = np.zeros_like(x)
#     for i in range(1,len(x)-1):
#         dudt[i] = D*(u[i+1]+u[i-1]-2*u[i])/dx**2
#     dudt[0]=0 #BCs
#     dudt[-1]=0
#     return dudt

# @jit
# def RHS_averaged(t,u,x):
#     dx = x[1]-x[0]
#     dudt = np.zeros_like(x)
#     for i in range(1,len(x)-1):
#         dudt[i] = Kappa*(u[i+1]+u[i-1]-2*u[i])/dx**2
#     dudt[0]=0 #BCs
#     dudt[-1]=0
#     return dudt


@jit
def initial_data(x):   
    return np.sin(np.pi*x)

#spacetime discretisation
x = np.linspace(0,1,2000)
t_max=1e-1
t_vals=np.linspace(0,t_max,10)

soln_full = scipy.integrate.solve_ivp(RHS_full,[0,t_max],initial_data(x),dense_output=True,args=(x,epsilon,))
# soln_naive = scipy.integrate.solve_ivp(RHS_no_advection,[0,t_max],initial_data(x),dense_output=True,args=(x,))
# soln_leading_average = scipy.integrate.solve_ivp(RHS_averaged,[0,t_max],initial_data(x),dense_output=True,args=(x,))

@jit
def du_0dx(x,t):
    return np.pi*np.cos(np.pi*x)*np.exp(-Kappa*np.pi**2*t)

# du0dx = np.zeros_like(soln_leading_average.sol(t_vals))
# for i in range(1,len(x)-1):
#     for j in range(len(t_vals)):
#         du0dx[i,j] = (soln_leading_average.sol(t_vals)[i+1,j]-soln_leading_average.sol(t_vals)[i-1,j])/(2*(x[1]-x[0]))
#         du0dx[0,j] = du0dx[1,j]
#         du0dx[len(x)-1,j] = du0dx[len(x)-2,j]


def antideriv_of_one_over_rho(y):
    return scipy.integrate.quad(one_over_rho,0,y)[0]

def chi(x,epsilon):
    y = np.remainder(x/epsilon,1)
    return -y + 1/z_bar*(antideriv_of_one_over_rho(y) - scipy.integrate.quad(antideriv_of_one_over_rho,0,1)[0])+1/2

def chi_func(x_array,epsilon):
    chi_array = np.zeros_like(x_array)
    for i,x in enumerate(x_array):
        chi_array[i] = chi(x,epsilon)
    return chi_array

def u_1(x,t,epsilon):
    if type(x)==np.ndarray:   
        return du_0dx(x,t)*chi_func(x,epsilon)
    elif type(x)==int or float or np.float64:
        return du_0dx(x,t)*chi(x,epsilon)
    else:
        raise Exception('Chi is broken')

# def u_1(epsilon):   
#     u1 = np.zeros_like(du0dx)
#     for i in range(1,len(x)-1):
#         for j in range(len(t_vals)):
#             u1[i,j]=chi(x[i],epsilon)*du0dx[i,j]
#             u1[0,j]=u1[len(x)-2,j] #periodic BCs
#             u1[len(x)-1,j]=u1[1,j]
#     return u1

#figsize=(fig_width_inches,fig_height_inches)
fig,ax = plt.subplots(1,3,figsize=(10,5))
plt.subplots_adjust(top=0.89, right=0.77)
plt.figure(2)
for t in t_vals:
    ax[0].plot(x,soln_full.sol(t),label=rf'$t= {np.round(t,3)}$')
    ax[2].plot(x,u_naive(x,t),label=rf'$t= {np.round(t,3)}$')
    ax[1].plot(x,u_0(x,t),label=rf'$t= {np.round(t,3)}$')
    # ax[2].plot(x,soln_naive.sol(t),label=rf'$t= {np.round(t,3)}$')
    # ax[1].plot(x,soln_leading_average.sol(t),label=rf'$t= {np.round(t,3)}$')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$u$')
ax[0].set_title('Full solution')
ax[0].grid()

ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$\tilde{u}$')
ax[1].set_title('Leading order averaged solution')
ax[1].grid()

ax[2].set_xlabel(r'$x$')
ax[2].set_ylabel(r'$u_0$')
ax[2].set_title('Naively averaged solution')
ax[2].grid()
ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax[0].set_position([0.06,0.11,0.22,0.78])
ax[1].set_position([0.35,0.11,0.22,0.78])
ax[2].set_position([0.64,0.11,0.22,0.78])
plt.suptitle(rf'$\varepsilon = {epsilon}$, $D = {D}$')
# ax[1,1].set_xlabel(r'$x$')
# ax[1,1].set_ylabel(r'$u_0+\varepsilon u_1$')
# ax[1,1].set_title('First order averaged solution')
# ax[1,1].legend(loc='center left', bbox_to_anchor=(1, 1.25))
# ax[1,1].grid()
# ax[1,1].set_position([0.52,0.1,0.33,0.33])
plt.savefig("adv-diff1fig2.pdf", format="pdf", bbox_inches="tight")


fig,ax = plt.subplots(2,2,figsize=(8,5))
fig.subplots_adjust(top=0.95)
plt.figure(3)
ax[0,0].plot(x,soln_full.sol(t_max),'-k',label=r'$u$')
ax[0,0].plot(x,u_naive(x,t_max),'-.',c='b',label=r'$\tilde{u}$')
# ax[0,0].plot(x,soln_leading_average.sol(t_max),'-.',c='b',label=r'$\tilde{u}$')
ax[0,0].set_xlabel(r'$x$')
ax[0,0].set_ylabel(r'$u$')
ax[0,0].grid()
ax[0,0].set_title('Full and naive solutions')
ax[0,0].legend()
ax[0,0].set_position([0.1,0.575,0.375,0.33])

ax[0,1].plot(x,soln_full.sol(t_max),'-k',label=r'$u$')
ax[0,1].plot(x,u_0(x,t_max),'-.r',label=r'$u_0$')
ax[0,1].plot(x,u_0(x,t_max) + epsilon*u_1(x,t_max,epsilon),'--b',label=r'$u_0 + \varepsilon u_1$')
# ax[0,1].plot(x,soln_leading_average.sol(t_max),'-.r',label=r'$u_0$')
# ax[0,1].plot(x,soln_leading_average.sol(t_max) + epsilon*u_1(epsilon)[:,-1],'--b',label=r'$u_0 + \varepsilon u_1$')
ax[0,1].set_xlabel(r'$x$')
ax[0,1].set_ylabel(r'$u$')
ax[0,1].grid()
ax[0,1].set_title('Full and average solutions')
ax[0,1].legend()
ax[0,1].set_position([0.6,0.575,0.375,0.33])

ax[1,0].plot(x,soln_full.sol(t_max) - u_0(x,t_max),'-r',label=r'$u - u_0$')
# ax[1,0].plot(x,soln_full.sol(t_max) - soln_leading_average.sol(t_max),'-r',label=r'$u - u_0$')
ax[1,0].set_xlabel(r'$x$')
ax[1,0].set_ylabel(r'$u - u_0$')
ax[1,0].grid()
ax[1,0].set_title('Error in leading order solution')
ax[1,0].set_position([0.1,0.1,0.375,0.33])

ax[1,1].plot(x,soln_full.sol(t_max) - u_0(x,t_max) - epsilon*u_1(x,t_max,epsilon),'--b',label=r'$u - (u_0 + \varepsilon u_1)$') 
# ax[1,1].plot(x,soln_full.sol(t_max) - soln_leading_average.sol(t_max) - epsilon*u_1(epsilon)[:,-1],'--b',label=r'$u - (u_0 + \varepsilon u_1)$')
ax[1,1].set_xlabel(r'$x$')
ax[1,1].set_ylabel(r'$u - (u_0+ \varepsilon u_1)$')
ax[1,1].grid()
ax[1,1].set_title('Error in first order solution')
ax[1,1].set_position([0.6,0.1,0.375,0.33])
plt.suptitle(rf'$t = {t_max}$, $\varepsilon = {epsilon}$, $D = {D}$')

plt.savefig("adv-diff1fig3.pdf", format="pdf", bbox_inches="tight")

def q(x,t,epsilon):
    return epsilon*(-u_1(1,t,epsilon)*x+u_1(0,t,epsilon)*(x-1))

fig,ax = plt.subplots(1,2,figsize=(fig_width_inches,fig_height_inches))
plt.figure(4)
ax[0].plot(x,soln_full.sol(t_max),'-k',label=r'$u$')
ax[0].plot(x,u_0(x,t_max) + epsilon*u_1(x,t_max,epsilon)+q(x,t_max,epsilon),'--b',label=r'$u_0 + \varepsilon u_1+q$')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$u$')
ax[0].grid()
ax[0].legend()
ax[0].set_title('Full and first order solutions')

ax[1].plot(x,soln_full.sol(t_max) - u_0(x,t_max) - epsilon*u_1(x,t_max,epsilon)-q(x,t_max,epsilon),'--b',label=r'$u - (u_0 + \varepsilon u_1 +q)$')
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$u - (u_0+ \varepsilon u_1+q)$')
ax[1].grid()
ax[1].set_title('Error in first order solution')
plt.suptitle(rf'$t = {t_max}$, $\varepsilon = {epsilon}$, $D = {D}$')
plt.tight_layout()
plt.savefig("adv-diff1fig5.pdf", format="pdf", bbox_inches="tight")


epsilons = 1/2**np.arange(2,7)
errors1 = np.zeros_like(epsilons)
errors2 = np.zeros_like(epsilons)
errors3 = np.zeros_like(epsilons)


for i, epsilon in enumerate(epsilons):
    x,dx = np.linspace(0,1,2001,retstep=True)
    soln_full = scipy.integrate.solve_ivp(RHS_full,[0,t_max],initial_data(x),dense_output=True,args=(x,epsilon,),rtol=1e-8,atol=1e-10)
    # soln_leading_average = scipy.integrate.solve_ivp(RHS_averaged,[0,t_max],initial_data(x),dense_output=True,args=(x,),rtol=1e-8,atol=1e-10)

    errors1[i] = np.linalg.norm(abs(soln_full.sol(t_max)- u_0(x,t_max)),1)*dx
    # errors1[i] = np.linalg.norm(abs(soln_full.sol(t_max)- soln_leading_average.sol(t_max)),1)*dx
    errors2[i] = np.linalg.norm(abs(soln_full.sol(t_max)- u_0(x,t_max)-epsilon*u_1(x,t_max,epsilon)),1)*dx
    errors3[i] = np.linalg.norm(abs(soln_full.sol(t_max)- u_0(x,t_max)-epsilon*u_1(x,t_max,epsilon)-q(x,t_max,epsilon)),1)*dx


errors1_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors1),1)
errors2_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors2),1)
errors3_polyfit_coeffs = np.polyfit(np.log(epsilons),np.log(errors3),1)

def trendline(data,x):
    return np.poly1d(data)(x)

plt.figure(5,figsize=(fig_width_inches,fig_height_inches))
plt.loglog(epsilons,np.exp(trendline(errors1_polyfit_coeffs,np.log(epsilons))),'-r',label=rf'$\propto \varepsilon^{{{np.around(errors1_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons,np.exp(trendline(errors2_polyfit_coeffs,np.log(epsilons))),'--',c='orange',label=rf'$\propto \varepsilon^{{{np.around(errors2_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons,errors1,'xk',label='Error at leading order')
plt.loglog(epsilons,errors2,'.b',label='Error at first order')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$\left\Vert\mathrm{Error}\right\Vert_{1}$')
plt.title(r'Error against $\varepsilon$')
plt.legend()
plt.savefig("adv-diff1fig4.pdf", format="pdf", bbox_inches="tight")

plt.figure(6,figsize=(fig_width_inches,fig_height_inches))
plt.loglog(epsilons,np.exp(trendline(errors3_polyfit_coeffs,np.log(epsilons))),'--',c='orange',label=rf'$\propto \varepsilon^{{{np.around(errors3_polyfit_coeffs[0],1)}}}$')
plt.loglog(epsilons,errors3,'.b',label='Error at first order with boundary correction')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$\left\Vert\mathrm{Error}\right\Vert_{1}$')
plt.title(r'Error against $\varepsilon$')
plt.legend()
plt.savefig("adv-diff1fig6.pdf", format="pdf", bbox_inches="tight")

plt.show()
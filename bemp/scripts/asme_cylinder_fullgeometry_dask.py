import numpy as np
import dask
from dask.distributed import Client
import bempp.api
import numpy as np
import dask
import time
from dask.distributed import Client, LocalCluster
import time
from dask.diagnostics import progress
import numba
import dask.array as da

#####################################
# ##################Predefine Funcs #
######################################



# create mk_potential
@dask.delayed
def create_Mk_pot(space,points_ff,k):
     Mk_pot = bempp.api.operators.potential.helmholtz.double_layer(space,
     points_ff, k)
     return Mk_pot


#create Lk_pot
@dask.delayed
def create_Lk_pot(space,points_ff,k):
     Lk_pot = bempp.api.operators.potential.helmholtz.single_layer(space,
     points_ff, k)
     return Lk_pot


# create identity operator
@dask.delayed
def create_identity(space):
     I   = bempp.api.operators.boundary.sparse.identity(space, space, space)
     return I

# create Lk
@dask.delayed
def create_Lk_boundary(space,k):
     Lk  = bempp.api.operators.boundary.helmholtz.single_layer(space,
     space, space, k)
     return Lk

# create Mk
@dask.delayed
def create_Mk_boundary(space,k):
     Mk  = bempp.api.operators.boundary.helmholtz.double_layer(space,
     space, space, k)
     return Mk

@dask.delayed
def create_theta(I):
     Theta = I.weak_form().A
     return Theta

@dask.delayed
def get_weak_form_Adjoint(input_obj):
     return input_obj.weak_form().A


@dask.delayed
def compute_space(grid):
     space = bempp.api.function_space(grid,"DP",0)
     return space



@numba.njit(fastmath=True)
def mat_inv(A):
     return np.linalg.inv(A)



def compute_weighting(ARMs,Theta):
     weighting = da.diag(da.transpose(ARMs) * Theta * ARMs)
     return weighting

def compute_radEff(S,rho0,c0,weighting):
     radEff = da.divide(S / (rho0*c0), weighting)  # [unitless]
     return radEff


#@dask.delayed
def compute_pfield(ARM,Z,space,points_ff,k):
     p_surf = Z2 @ ARM
     Mk_pot = bempp.api.operators.potential.helmholtz.double_layer(space,
     points_ff, k)
     Lk_pot = bempp.api.operators.potential.helmholtz.single_layer(space,
     points_ff, k)
     grid_fun_v = bempp.api.GridFunction(space, coefficients=ARM)
     grid_fun_p = bempp.api.GridFunction(space, coefficients=p_surf)
     phi_field = Mk_pot*grid_fun_p / (1j*rho0*omega) - Lk_pot*grid_fun_v
     p_field = 1j*rho0*omega*phi_field # [Pa] far-field pressure


#@dask.delayed
def make_grid_fun_v(space,ARM):
     grid_v = bempp.api.GridFunction(space, coefficients=ARM)
     return grid_v


#@dask.delayed
def make_grid_fun_p(space,p_surf):
     grid_fun_p = bempp.api.GridFunction(space, coefficients=p_surf)
     return grid_fun_p



@numba.njit
def inverse_mult(H,G):
     v = np.linalg.inv(H) * G
     return v.T

##################
# Define variables
##################



# clust = LocalCluster(n_workers=1,threads_per_worker=1)


ka = np.arange(0.1, 10.1, 0.1)
a = 1
r = 1000
c = 1500
rho = 1000
k = 1
omega = c*k
rho0 = 1000
p_field = np.zeros(ka.shape, dtype='complex')


Rff = 10000.0 # [m] distance to far-field
theta = np.linspace(0,2*np.pi,360)
x,y = Rff*np.cos(theta),Rff*np.sin(theta)
z = np.zeros_like(x)

points_ff = np.row_stack((x,y,z))




################
# Startup Dask #
################

clust = LocalCluster(n_workers=8,threads_per_worker=1)
client = Client(clust)


#####################
# build computation #
#####################

grid = bempp.api.grid.io.import_grid(r"C:\Users\BlubaughFC\Documents\SideProjects\bemppdocker\bempp\meshes\asme_cylinder_3d_bem_mesh_1000Hz_LinearElements.bdf")

# grid_future = client.scatter(grid,broadcast=True)

space = compute_space(grid)
Mk = create_Mk_boundary(space,k)
Lk = create_Lk_boundary(space,k)
I = create_identity(space)

Theta = create_theta(I)

lhs = Mk-0.5*I
rhs = Lk

# compute the H and G arrays

H = da.from_delayed(get_weak_form_Adjoint(lhs),shape=(grid.number_of_elements,grid.number_of_elements),dtype='complex128').rechunk()/(1j*omega*rho0)


G = da.from_delayed(get_weak_form_Adjoint(rhs),shape=(grid.number_of_elements,grid.number_of_elements),dtype='complex128').rechunk()

Theta = da.from_delayed(Theta,shape=(grid.number_of_elements,grid.number_of_elements),dtype='float64').rechunk()


%time Z = np.transpose(np.linalg.inv(H)*G)*Theta

Z = da.transpose(da.linalg.inv(H)*G)*Theta

# compute ARMs
ZR = da.real(Z)
ZRs = 0.5 * (ZR + da.transpose(ZR))
ZRs = ZRs.rechunk(chunks=(-1,1))
ARMs,S,vh = da.linalg.svd(ZRs)
# ARMs2,S,vh = dask.compute(ARMs,S,vh)


weighting = compute_weighting(ARMs,Theta)

radEff = compute_radEff(S,rho0,c,weighting)






# compute some results....hoepfully it doesn't crash

# Calculate a bunch of radiation modes:
Mk_pot = create_Mk_pot(space, points_ff, k)
Lk_pot = create_Lk_pot(space, points_ff, k)

results_list = []
for i in range(160):
     p_surf = da.dot(Z,ARMs[:,i])

     grid_fun_v = make_grid_fun_v(space,ARMs[:,i])
     grid_fun_p = make_grid_fun_p(space,p_surf)


     phi_field = Mk_pot*grid_fun_p / (1j*rho0*omega) - Lk_pot*grid_fun_v
     results_list.append(phi_field)

tic = time.perf_counter()
vv = client.gather(client.compute(results_list))
toc = time.perf_counter()
print(toc-tic)



@bempp.api.complex_callable
def dipole_source(x,n,domain_index,result):
# all points positive x will be 1, negative x will be -1
result[0] = x[0]
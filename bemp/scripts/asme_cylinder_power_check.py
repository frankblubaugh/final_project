

# Import packages
import bempp.api
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy as scp
import h5py


###################################
#        Define Parameters        #
###################################
f_min = 100.0
f_step = 50.0
f_max = 1000.0
freq = np.arange(start=f_min, stop=(f_step+f_max), step=f_step)
Nfreq = freq.size

N_RS = 10 # number of random seeds to consider

c0 = 1500.0 # [m/s]
rho0 = 1000.0 # [kg/m^3]
v0 = 1.0e-3 # [m/s] velocity amplitude

# For random velocity distribution
L = 10
M = 10
N = 20
Lx = 2.
Ly = 2.
Lz = 12.

###################################
#           Create Grid           #
###################################
grid = bempp.api.grid.io.import_grid('../meshes/asme_cylinder_3d_bem_mesh_1000Hz_LinearElements.nas')

# Uncomment the following line to plot the grid
#grid.plot()

# Get element areas
elemAreas = grid.volumes
Theta = np.diag(elemAreas)

# Print details about the grid
print("Number of edges = {0}".format(grid.number_of_edges))
print("Number of elements = {0}".format(grid.number_of_elements))
print("Number of vertices = {0}".format(grid.number_of_vertices))

# Use piecewise constant functions
space = bempp.api.function_space(grid,"DP",0)


###################################
#      Define Grid Functions      #
###################################
@bempp.api.real_callable
def f_uniform(x, n, domain_index, result):
    result[0] = v0

grid_fun_uniform = bempp.api.GridFunction(space, fun=f_uniform)

# Load matrix for random velocity distribution
f = h5py.File('../spherePseudorandomVelocityDistribution.mat','r')
A_lmn = list(f['A_lmn'])
A_lmn = np.array(A_lmn)
A_lmn = np.swapaxes(A_lmn, 0,1)
A_lmn = np.swapaxes(A_lmn, 1,2)
A_lmn = np.swapaxes(A_lmn, 2,3)
A_lmn = np.swapaxes(A_lmn, 0,1)
A_lmn = np.swapaxes(A_lmn, 1,2)
A_lmn = np.swapaxes(A_lmn, 0,1)
print(A_lmn.shape)

###################################
#         Frequency Sweep         #
###################################
power_uniform_Z = np.zeros(Nfreq) # solve power using impedance matrix
power_uniform_mass = np.zeros(Nfreq) # solve power using Theta matrix (element areas)
power_random_Z = np.zeros([Nfreq, N_RS]) # solve power using impedance matrix
power_random_mass = np.zeros([Nfreq, N_RS]) # solve power using Theta matrix (element areas)
v0_rand = np.zeros([grid.number_of_elements, N_RS])
print("Starting frequency sweep")

for iFreq in range(0,Nfreq):
    print("Frequency step {0}".format(iFreq))
    omega = freq[iFreq] * 2.0*np.pi # [rad/s] angular frequency
    k = omega / c0 # [1/m] wavenumber
    
    # Build the boundary operators
    I   = bempp.api.operators.boundary.sparse.identity(space, space, space)
    Lk  = bempp.api.operators.boundary.helmholtz.single_layer(space, space, space, k)
    Mk  = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, k)

    # Apply elementary direct method
    lhs = Mk - 0.5 * I
    rhs = Lk
    
    ####################################
    #     Compute impedance matrix     #
    ####################################
    H = lhs.weak_form().A/(1j*omega*rho0)
    G = rhs.weak_form().A
    Z = np.transpose(np.linalg.inv(H)*G) * Theta
    
    ####################################
    #          Random velocity         #
    ####################################
    for iRS in range(0,N_RS):
        @bempp.api.real_callable
        def f_random(x, normal, domain_index, result):
            result[0] = 0
            for l in range(1,L+1):
                for m in range(1,M+1):
                    for n in range(1,N+1):
                        result[0] += A_lmn[l-1,m-1,n-1,iRS]*v0 * np.sin(l*np.pi*x[0]/Lx) * np.sin(m*np.pi*x[1]/Ly) * np.sin(n*np.pi*x[2]/Lz)


        grid_fun_rand = bempp.api.GridFunction(space, fun=f_random)
        phi_fun, info = bempp.api.linalg.gmres(lhs, rhs*grid_fun_rand, tol=1E-7)
    
        v = np.transpose(grid_fun_rand.evaluate_on_element_centers()) # column vector
        p = 1j*rho0*omega*np.transpose(phi_fun.evaluate_on_element_centers()) # column vector
    
        power_cur = 0.5 * np.real( np.matmul(np.matmul(np.transpose(p), Theta), np.conjugate(v)) )
        power_random_mass[iFreq,iRS] = power_cur.item()
        if power_cur.item()==0:
            print("Power is zero using Theta")

        power_cur = 0.5 * np.real( np.matmul(np.matmul(np.transpose(v), Z), np.conjugate(v)) )
        power_random_Z[iFreq,iRS] = power_cur.item()
        
    ####################################
    #         Uniform velocity         #
    ####################################
    phi_fun, info = bempp.api.linalg.gmres(lhs, rhs*grid_fun_uniform, tol=1E-7)
    
    v = np.transpose(grid_fun_uniform.evaluate_on_element_centers()) # column vector
    p = 1j*rho0*omega*np.transpose(phi_fun.evaluate_on_element_centers()) # column vector
    
    power_cur = 0.5 * np.real( np.matmul(np.matmul(np.transpose(p), Theta), np.conjugate(v)) )
    power_uniform_mass[iFreq] = power_cur.item()
    
    power_cur = 0.5 * np.real( np.matmul(np.matmul(np.transpose(v), Z), np.conjugate(v)) )
    power_uniform_Z[iFreq] = power_cur.item()

    


####################################
#             Save Data            #
####################################
bempp_data = {'freq':freq, 'rho0':rho0, 'c0':c0, 'element_centers': grid.centroids, 'elements':grid.elements, 
      'vertices':grid.vertices, 'volumes':grid.volumes, 'element_edges':grid.element_edges,'edges':grid.edges,
     'power_uniform_mass':power_uniform_mass, 'power_uniform_Z':power_uniform_Z,
     'power_random_mass':power_random_mass, 'power_random_Z':power_random_Z}
sio.savemat('bempp_asme_cylinder_power_check.mat', bempp_data)

print("Done.")
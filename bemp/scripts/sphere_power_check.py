

# Import packages
import bempp.api
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy as scp
import h5py
import math


###################################
#        Define Parameters        #
###################################
ka_min = 0.1
ka_max = 10.0
Nfreq = 51
ka_range = np.logspace(np.log10(ka_min), np.log10(ka_max), num=Nfreq)

# Add zeros of spherical Bessel function to frequencies
ka_range = np.append(ka_range, [np.pi, 2*np.pi, 3*np.pi, 4.493, 5.763, 6.988, 8.183, 7.7725, 9.095, 2.082, 3.342, 4.514, 5.647])
ka_range = np.sort(ka_range)
Nfreq = ka_range.size
ka_max = np.max(ka_range)

N_RS = 5 # number of random seeds to consider

a = 1.0 # [m]
c0 = 1500.0 # [m/s]
rho0 = 1000.0 # [kg/m^3]
v0 = 1.0e-3 # [m/s] velocity amplitude

k_max = ka_max/a # [1/m]
omega_max = c0*k_max # [rad/s]
lambda_max = 2.0*np.pi/k_max # [m] shortest wavelength
elemSize = lambda_max/12.0 # [m] element size

print("Element size: {0} m".format(elemSize))

# For random velocity distribution
L = 6
M = 6
N = 6
Lx = 2*a
Ly = 2*a
Lz = 2*a

###################################
#           Create Grid           #
###################################
grid = bempp.api.shapes.sphere(h=elemSize)

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



# Define elastic parameters
thickness = 0.01 # [m] shell thickness
F = 1.0 # [N] force amplitude
rhos = 7850. # [kg/m^3] structure density
E = 2E11 # [Pa] Young's modulus
nu=0.3 # Poison ratio of the sphere

cp = np.sqrt(E / ((1-nu**2)*rhos))
beta = 1/np.sqrt(12.0) * thickness/a

N_point_force = 8 # number of terms to include in point force summation

point_source_coeff = np.zeros((Nfreq,grid.number_of_elements), dtype=complex)
xec = grid.centroids[:,0]
yec = grid.centroids[:,1]
zec = grid.centroids[:,2]
theta = np.arctan2(np.sqrt(xec**2+yec**2), zec) # [rad] inclination angle
for iFreq in range(0,Nfreq):
    k = ka_range[iFreq]/a # [1/m] wavenumber
    omega = k * c0 # [rad/s] angular frequency
    Omega = omega*a/cp # [nondimensional] frequency
    for n in range(0,N_point_force+1):
        lambda_n = n*(n+1)

        hn_ka      = scp.special.spherical_jn(n,ka_range[iFreq]) + 1j*scp.special.spherical_yn(n,ka_range[iFreq])
        hnPrime_ka = scp.special.spherical_jn(n,ka_range[iFreq],1) + 1j*scp.special.spherical_yn(n,ka_range[iFreq],1)

        z_n = 1j*rho0*c0 * hn_ka / hnPrime_ka # Eq. 6.29 in Junger & Feit

        termA = 1.0 # term proportional to Omega^4 in Eq. 7.114 in Junger & Feit
        termB = -(1+3*nu+lambda_n-beta**2*(1-nu-lambda_n**2-nu*lambda_n)) # term proportional to Omega^2
        termC = (lambda_n-2)*(1-nu**2) + beta**2*(lambda_n**3-4*lambda_n**2+lambda_n*(5-nu**2)-2*(1-nu**2)) # term proportional to 1
        Omega2_n1 =  (-termB + np.sqrt(termB**2-4*termA*termC))/(2*termA)
        Omega2_n2 =  (-termB - np.sqrt(termB**2-4*termA*termC))/(2*termA)
        Z_n = -1j*thickness/a*rhos*cp * ( (Omega**2-Omega2_n1)*(Omega**2-Omega2_n2)) / (Omega**3-Omega*(n**2+n-1+nu))
        
        Pn = scp.special.eval_legendre(n,np.cos(theta))
        temp = F/(4*np.pi*a**2) * (2*n+1)/(Z_n+z_n) * Pn # [m/s] Eq. 9.13 in Junger & Feit
        point_source_coeff[iFreq,:] += temp
print("Done calculating velocity distribution for point force")





###################################
#         Frequency Sweep         #
###################################
power_uniform_Z    = np.zeros(Nfreq) # solve power using impedance matrix
power_uniform_mass = np.zeros(Nfreq) # solve power using Theta matrix (element areas)
power_random_Z     = np.zeros([Nfreq, N_RS]) # solve power using impedance matrix
power_random_mass  = np.zeros([Nfreq, N_RS]) # solve power using Theta matrix (element areas)
power_point_mass   = np.zeros(Nfreq) # solve power using Theta matrix (element areas)
power_point_Z      = np.zeros(Nfreq) # solve power using Theta matrix (element areas)
v0_rand = np.zeros([grid.number_of_elements, N_RS])
print("Starting frequency sweep")

for iFreq in range(0,Nfreq):
    print("Frequency step " + str(iFreq) + " of " + str(Nfreq))
    k = ka_range[iFreq]/a # [1/m] wavenumber
    omega = k * c0 # [rad/s] angular frequency
    
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
    #        Point force velocity      #
    ####################################
    grid_fun_point = bempp.api.GridFunction(space, coefficients=point_source_coeff[iFreq,:])

    phi_fun, info = bempp.api.linalg.gmres(lhs, rhs*grid_fun_point, tol=1E-7)
    
    v = np.transpose(grid_fun_point.evaluate_on_element_centers()) # column vector
    p = 1j*rho0*omega*np.transpose(phi_fun.evaluate_on_element_centers()) # column vector
    
    power_cur = 0.5 * np.real( np.matmul(np.matmul(np.transpose(p), Theta), np.conjugate(v)) )
    power_point_mass[iFreq] = power_cur.item()
    
    power_cur = 0.5 * np.real( np.matmul(np.matmul(np.transpose(v), Z), np.conjugate(v)) )
    power_point_Z[iFreq] = power_cur.item()

####################################
#             Save Data            #
####################################
bempp_data = {'ka':ka_range, 'a':a, 'rho0':rho0, 'c0':c0, 'element_centers': grid.centroids, 'elements':grid.elements, 
      'vertices':grid.vertices, 'volumes':grid.volumes, 'element_edges':grid.element_edges,'edges':grid.edges,
     'power_uniform_mass':power_uniform_mass, 'power_uniform_Z':power_uniform_Z,
     'power_random_mass':power_random_mass, 'power_random_Z':power_random_Z,
     "power_point_mass":power_point_mass, "power_point_Z":power_point_Z,
     "N_point_force":N_point_force}
sio.savemat('bempp_sphere_power_check_fromScript.mat', bempp_data)

print("Done.")
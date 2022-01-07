import bempp.api
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio



######################################
#             Parameters             #
######################################
ka_min = 0.1
ka_max = 5
Nfreq = 60

a = 1 # [m]
c0 = 1500 # [m/s]
rho0 = 1000 # [kg/m^3]

k_max = ka_max/a # [1/m]
omega_max = c0*k_max
lambda_max = 2*np.pi/k_max # [m] shortest wavelength
elemSize = lambda_max/20 # [m] element size

ka_range = np.logspace(np.log10(ka_min), np.log10(ka_max), num=Nfreq)

######################################
#             Create Grid            #
######################################
grid = bempp.api.shapes.cube(length=a, h=elemSize)

# Uncomment the following line to plot the grid
#grid.plot()

# Print out details about the grid
print("Number of edges = {0}".format(grid.number_of_edges))
print("Number of elements = {0}".format(grid.number_of_elements))
print("Number of vertices = {0}".format(grid.number_of_vertices))


######################################
#           Frequency sweep          #
######################################

# Use piecewise constant functions
space = bempp.api.function_space(grid,"DP",0)

for iFreq in range(0,Nfreq):
    
    k = ka_range[iFreq]/a       # [1/m] wavenumber
    omega = k * c0              # [rad/s] angular frequency
    
    # Build the boundary operators
    I   = bempp.api.operators.boundary.sparse.identity(space, space, space)
    Lk  = bempp.api.operators.boundary.helmholtz.single_layer(space, space, space, k)
    Mk  = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, k)

    # Apply elementary direct method
    lhs = Mk - 0.5 * I
    rhs = Lk

    # Discretize the operators by computing the weak form
    H = lhs.weak_form().A / (1j*omega*rho0)
    G = rhs.weak_form().A

    # Compute the impedance matrix
    Z = np.linalg.inv(H)*G
    
    # Save data to MAT file
    bempp_data = {'k': k, 'Z': Z, 'element_centers': grid.centroids}
    filename = "bempp_cube_ka{:06.4f}.mat".format(ka_range[iFreq])
    sio.savemat(filename, bempp_data)
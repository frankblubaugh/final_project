# Frequency sweep of ASME Cylinder. Mesh imported from PLY binary file created in COMSOL

import bempp.api
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

######################################
#             Parameters             #
######################################
freq_min = 50.0 # [Hz]
freq_max = 1000.0 # [Hz]
Nfreq = 60

c0 = 1500.0 # [m/s]
rho0 = 1000.0 # [kg/m^3]

freq_range = np.logspace(np.log10(freq_min), np.log10(freq_max), num=Nfreq)

######################################
#             Create Grid            #
######################################
grid = bempp.api.grid.io.import_grid('../meshes/asme_cylinder_3d_bem_mesh_500HzLinear.ply')

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
    
    omega = 2.0 * np.pi * freq_range[iFreq] # [rad/s] angular frequnecy
    k = omega / c0                          # [1/m] wavenumber
    
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
    bempp_data = {'k': k, 'Z': Z, 'element_centers': grid.centroids, 'elements':grid.elements, 'vertices':grid.vertices, 'volumes':grid.volumes, 'element_edges':grid.element_edges,'edges':grid.edges, 'freq':freq_range[iFreq],
 "H":H, "G":G}
    filename = "bempp_asme_cylinder_{:06.1f}Hz.mat".format(freq_range[iFreq])
    sio.savemat(filename, bempp_data)
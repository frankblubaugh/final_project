#!/usr/bin/env python
# coding: utf-8

# # Plate modal analysis
# 
# This modal analysis of a plate works and draws off of a gmsh value to produce the modal analysis. This was confirmed by modeling the NAFEMS plate from the UK in 1990 and confirming the results

# # Notes
# - Gmsh must be installed and has been added to the docker file
# - Only really works in dolfin not dolfinx. this is due to the lack of a good assembly with the boundary conditions needed for the problem

# In[1]:


# import stuff


from dolfin import *
import dolfin
comm = dolfin.cpp.MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()
print(f"rank: {rank}; size:{size}")
import petsc4py
import numpy as np
import time
import matplotlib.pyplot as plt
# %matplotlib notebook
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True


# In[2]:




# load in gmsh
mesh = Mesh(comm,"../mesh/extruded_box.xml")
# # In[2]:

E, nu = Constant(200e9), Constant(0.23)
rho = Constant(8000)


# In[ ]:





# In[3]:



# Lame coefficient for constitutive relation
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

def eps(v):
    return sym(grad(v))

def sigma(v):
    dim = v.geometric_dimension()
    return 2.0*mu*eps(v) + lmbda*tr(eps(v))*Identity(dim)


# In[4]:



V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)
u_ = TrialFunction(V)
du = TestFunction(V)


# ## Notes on Boundary Condition
# 
# The simply supported boundary condition (Z =0) is applied to the third subspace of the function space. V.sub(2)

# In[5]:



def simply_supported(x,on_boundary):
    return on_boundary and (near(x[0],0.0) | near(x[0],10.0)| near(x[1],0.0)| near(x[1],10.0))



bc = DirichletBC(V.sub(2),Constant((0.)), simply_supported)


# ## Notes on building system
# There are known symmetry issues with building the system with the K matrix. This solves them for uknown reasons

# In[6]:




k_form = inner(sigma(du),eps(u_))*dx
l_form = Constant(1.)*u_[0]*dx
K = PETScMatrix(comm)
b = PETScVector(comm)
assemble_system(k_form, l_form,bc, A_tensor=K, b_tensor=b)


# In[7]:



m_form = rho*dot(du,u_)*dx
M = PETScMatrix(comm)
assemble(m_form, tensor=M)
bc.zero(M)


# In[8]:




eigensolver = SLEPcEigenSolver(K,M)
eigensolver.parameters['problem_type'] = 'gen_hermitian'
eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
eigensolver.parameters['spectral_shift'] = 100.


# In[9]:



N_eig = 50   # number of eigenvalues
print("Computing {} first eigenvalues...".format(N_eig))
start = time.perf_counter()
eigensolver.solve(N_eig)
print(f"elapsed time: {time.perf_counter()-start}")


# In[10]:



print(f"N Solutions Found: {eigensolver.get_number_converged()}")
if rank == 0:
    for i in range(N_eig):
        # Extract eigenpair
        r, c, rx, cx = eigensolver.get_eigenpair(i)
        # 3D eigenfrequency
        freq_2D = sqrt(r)/2/pi
        print(f"Solid FEM: {freq_2D:.3f} Hz")


# In[11]:


eigenmodes = []

for i in range(N_eig):
    # Extract eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)

    # 3D eigenfrequency
    freq_3D = sqrt(r)/2/pi

   
    # Initialize function and assign eigenvector
    eigenmode = Function(V,name="Eigenvector "+str(i))
    eigenmode.vector()[:] = rx

    eigenmodes.append(eigenmode)
#     file_results.write(eigenmode,0.)
    


# In[12]:


zz = eigenmodes[3]


# In[13]:


# visualie the data by exporting to file


# file_results = XDMFFile("../data/modal_analysis.xdmf")
# file_results.parameters["flush_output"] = True
# file_results.parameters["functions_share_mesh"] = True
# file_results.write(eigenmode, 0.)


# # Visualization
# ## Notes
# - Ne3eds to have vedo installed

# In[15]:


from vedo.dolfin import *
from vedo import Box


# In[16]:


# add a frame box
box = Box(length=1, width=1, height=1).pos(0.5,0,0).wireframe()


# In[ ]:


plot(zz,box,interactive=False)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dolfin import *
import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD


L, B, H = 10,10,1

Nx = 16
Ny = int(B/L*Nx)+1
Nz = int(H/L*Nx)+1

mesh = BoxMesh(Point(0.,0.,0.),Point(L,B,H),32,32,3)

# # In[2]:


E, nu = Constant(200e9), Constant(0.23)
rho = Constant(8000)

# Lame coefficient for constitutive relation
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

def eps(v):
    return sym(grad(v))

def sigma(v):
    dim = v.geometric_dimension()
    return 2.0*mu*eps(v) + lmbda*tr(eps(v))*Identity(dim)


# In[3]:


V = VectorFunctionSpace(mesh, 'Lagrange', degree=2)
u_ = TrialFunction(V)
du = TestFunction(V)

##################
# DEFINE BCS #####
##################
def all_boundary(x,on_boundary):
	return on_boundary

def left(x, on_boundary):
    return near(x[0],2.)

def right(x, on_boundary):
    return on_boundary and near(x[0], 10.0)

def bottom(x, on_boundary):
    return on_boundary and near(x[1], 0.0)

def top(x, on_boundary):
    return on_boundary and near(x[1], 10.0)


def all_clamped(x,on_boundary):
    return on_boundary and (near(x[0],0.0) | near(x[0],10.0)| near(x[1],0.0)| near(x[1],10.0))

# bcs = [#DirichletBC(V, Constant((0.,0.,0.)), all_boundary)]
#  DirichletBC(V, Constant((0.,0.,0.)), left),
#   DirichletBC(V, Constant((0.,0.,0.)), right),
#    DirichletBC(V, Constant((0.,0.,0.)), bottom),
#     DirichletBC(V, Constant((0.,0.,0.)), top)]


bc = DirichletBC(V.sub(2),Constant((0.)), all_clamped)


# In[4]:


k_form = inner(sigma(du),eps(u_))*dx
l_form = Constant(1.)*u_[0]*dx
K = PETScMatrix()
b = PETScVector()
assemble_system(k_form, l_form,bc, A_tensor=K, b_tensor=b)

m_form = rho*dot(du,u_)*dx
M = PETScMatrix()
assemble(m_form, tensor=M)
bc.zero(M)

# In[5]:


eigensolver = SLEPcEigenSolver(K,M)
# eigensolver.set_operators()
eigensolver.parameters['problem_type'] = 'gen_hermitian'
eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
eigensolver.parameters['spectral_shift'] = 100.

# eigensolver.parameters["spectrum"] = "smallest magnitude"



# In[ ]:


N_eig = 50   # number of eigenvalues
print("Computing {} first eigenvalues...".format(N_eig))
start = time.perf_counter()
eigensolver.solve(N_eig)
print(f"elapsed time: {time.perf_counter()-start}")

# In[18]:


print(f"N Solutions Found: {eigensolver.get_number_converged()}")


# In[11]:




for i in range(N_eig):
    # Extract eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)

    # 3D eigenfrequency
    freq_2D = sqrt(r)/2/pi
    # Beam eigenfrequency
    # if i % 2 == 0: # exact solution should correspond to weak axis bending
    #     I_bend = H*B**3/12.
    # else:          #exact solution should correspond to strong axis bending
    #     I_bend = B*H**3/12.
    # freq_beam = alpha(i/2)**2*sqrt(float(E)*I_bend/(float(rho)*B*H*L**4))/2/pi
    # print(f" Solid FEM: {freq_3D:.3f} Hz; Beam Theory :{freq_beam:.3f} Hz")
    print(f"Solid FEM: {freq_2D:.3f} Hz")


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dolfin import *
import dolfin
import numpy as np
import time
from mpi4py import MPI

comm = dolfin.cpp.MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()
print(f"rank: {rank}; size:{size}")
if rank==0:
    print('first worker')
    # load in gmsh
    mesh = Mesh(comm,"mesh/extruded_box.xml")

    # # In[2]:
    print('loaded mesh')

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


    V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)
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


    def simply_supported(x,on_boundary):
        return on_boundary and (near(x[0],0.0) | near(x[0],10.0)| near(x[1],0.0)| near(x[1],10.0))



    bc = DirichletBC(V.sub(2),Constant((0.)), simply_supported)




    k_form = inner(sigma(du),eps(u_))*dx
    l_form = Constant(1.)*u_[0]*dx
    K = PETScMatrix()
    b = PETScVector()
    assemble_system(k_form, l_form,bc, A_tensor=K, b_tensor=b)

    m_form = rho*dot(du,u_)*dx
    M = PETScMatrix()
    assemble(m_form, tensor=M)
    bc.zero(M)



    eigensolver = SLEPcEigenSolver(K,M)
    eigensolver.parameters['problem_type'] = 'gen_hermitian'
    eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
    eigensolver.parameters['spectral_shift'] = 100.


    N_eig = 50   # number of eigenvalues
    print("Computing {} first eigenvalues...".format(N_eig))
    start = time.perf_counter()
    eigensolver.solve(N_eig)
    print(f"elapsed time: {time.perf_counter()-start}")

    # In[18]:


    print(f"N Solutions Found: {eigensolver.get_number_converged()}")

    for i in range(N_eig):
        # Extract eigenpair
        r, c, rx, cx = eigensolver.get_eigenpair(i)
        # 3D eigenfrequency
        freq_2D = sqrt(r)/2/pi
        # print(f"Solid FEM: {freq_2D:.3f} Hz")

else:
    req = comm.irecv(source=0,tag=11)



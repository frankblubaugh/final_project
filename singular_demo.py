
from dolfin import *
import matplotlib.pyplot as plt
import time
start = time.perf_counter()
# Then, we check that dolfin is configured with the backend called
# PETSc, since it provides us with a wide range of methods used by
# :py:class:`KrylovSolver <dolfin.cpp.la.KrylovSolver>`. We set PETSc as
# our backend for linear algebra::

# Test for PETSc
if not has_linear_algebra_backend("PETSc"):
    info("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

parameters["linear_algebra_backend"] = "PETSc"

# We begin by defining a mesh of the domain and a finite element
# function space :math:`V` relative to this mesh. We use a built-in mesh
# provided by the class :py:class:`UnitSquareMesh
# <dolfin.cpp.mesh.UnitSquareMesh>`. In order to create a mesh
# consisting of :math:`64 \times 64` squares with each square divided
# into two triangles, we do as follows: ::

# Create mesh and define function space
mesh = UnitSquareMesh(64, 64)
V = FunctionSpace(mesh, "CG", 1)

# Now, we need to specify the trial functions (the unknowns) and the
# test functions on the space :math:`V`. This can be done using a
# :py:class:`TrialFunction <dolfin.functions.function.TrialFunction>`
# and a :py:class:`TestFunction
# <dolfin.functions.function.TrialFunction>` as follows: ::

u = TrialFunction(V)
v = TestFunction(V)

# Further, the source :math:`f` and the boundary normal derivative
# :math:`g` are involved in the variational forms, and hence we must
# specify these. Both :math:`f` and :math:`g` are given by simple
# mathematical formulas, and can be easily declared using the
# :py:class:`Expression <dolfin.functions.expression.Expression>`
# class. Note that the strings defining f and g use C++ syntax since,
# for efficiency, DOLFIN will generate and compile C++ code for these
# expressions at run-time. ::

f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("-sin(5*x[0])", degree=2)

# With :math:`u,v,f` and :math:`g`, we can write down the bilinear form
# :math:`a` and the linear form :math:`L` (using UFL operators). ::

a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# In order to transform our variational problem into a linear system we
# need to assemble the coefficient matrix ``A`` and the right-side
# vector ``b``. We do this using the function :py:meth:`assemble
# <dolfin.cpp.fem.Assembler.assemble>`: ::

# Assemble system
A = assemble(a)
b = assemble(L)

# We specify a Vector for storing the result by defining a
# :py:class:`Function <dolfin.cpp.function.Function>`. ::

# Solution Function
u = Function(V)

# Next, we specify the iterative solver we want to use, in this case a
# :py:class:`PETScKrylovSolver <dolfin.cpp.la.PETScKrylovSolver>` with
# the conjugate gradient (CG) method, and attach the matrix operator to
# the solver. ::

# Create Krylov solver
solver = PETScKrylovSolver("cg")
solver.set_operator(A)

# We impose our additional constraint by removing the null space
# component from the solution vector. In order to do this we need a
# basis for the null space. This is done by creating a vector that spans
# the null space, and then defining a basis from it. The basis is then
# attached to the matrix ``A`` as its null space. ::

# Create vector that spans the null space and normalize
null_vec = Vector(u.vector())
V.dofmap().set(null_vec, 1.0)
null_vec *= 1.0/null_vec.norm("l2")

# Create null space basis object and attach to PETSc matrix
null_space = VectorSpaceBasis([null_vec])
as_backend_type(A).set_nullspace(null_space)

# Orthogonalization of ``b`` with respect to the null space makes sure
# that it doesn't contain any component in the null space. ::

null_space.orthogonalize(b);

# Finally we are able to solve our linear system ::

solver.solve(u.vector(), b)
print(time.perf_counter()-start)
# and plot the solution ::

plot(u)
plt.show()

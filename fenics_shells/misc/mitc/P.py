import sys
from dolfin import *

args = [sys.argv[0]] + """
--petsc.ksp_monitor_true_residual
--petsc.ksp_type preonly
--petsc.pc_type lu
--petsc.pc_mat_solver_package mumps 
""".split()
parameters.parse(args)

mesh = UnitSquareMesh(50, 50, "crossed")
mesh.init()
R = VectorFunctionSpace(mesh, "CG", 2)
V_3 = FunctionSpace(mesh, "CG", 1)
RR = FunctionSpace(mesh, "N1curl", 1)

import sys; sys.path.append("../../")
from fenics_shells.analytical.lovadina_clamped import Loading, Rotation, Displacement
E = 10920.0
nu = 0.3
tv = 10.0**-5
kappa = 5.0/6.0

theta_e = Rotation()
z_e = Displacement(t=tv, nu=nu)
f = Loading(E=E, nu=nu)

E = Constant(E)
nu = Constant(nu)
t = Constant(tv)
kappa = Constant(kappa)

e = lambda theta: sym(grad(theta))
B = lambda e: (E/(12.0*(1.0 - nu**2)))*((1.0 - nu)*e + nu*tr(e)*Identity(2))
F = (E*kappa*t**-2)/(2.0*(1.0 + nu))

U = MixedFunctionSpace([R, V_3])
U_RR = MixedFunctionSpace([R, V_3, RR])

# Full problem space
r, z, rr = TrialFunctions(U_RR)
r_t, z_t, rr_t = TestFunctions(U_RR)

n = FacetNormal(mesh)
t = as_vector((-n[1], n[0]))

# Want to apply projection operator a_p
dSp = Measure('dS', metadata={'quadrature_degree' : 1})
dsp = Measure('ds', metadata={'quadrature_degree' : 1})
area = FacetArea(mesh)
a_10 = (area*inner(r, t)*inner(rr_t, t))("+")*dSp + (area*inner(r, t)*inner(rr_t, t))*dsp 
# Note that I do not explicitly assemble the inner(grad(z), grad(z_t))*dx term here
# With the current assembler code we end up with the inner(grad(z), grad(z_t))*dx term being
# obliterated by the action of the projection operator. We probably need some identity terms
# in a_p but I am unsure as to how to get FFC to generate them.
a_01 = F*(-inner(rr, grad(z_t)) - inner(grad(z), rr_t) + inner(rr, rr_t))*dx 
a_11 = (area*inner(rr, t)*inner(rr_t, t))("+")*dSp + area*inner(rr, t)*inner(rr_t, t)*dsp

# Regardless, I do assemble that 'lost' term here.
# Assemble bending and inner(grad(z), grad(z_t))*dx term on standard spaces
r, z = TrialFunctions(U)
r_t, z_t = TestFunctions(U)
a_00 = inner(B(e(r)), e(r_t))*dx + F*inner(grad(z), grad(z_t))*dx

# Boundary conditions defined on standard spaces
def all_boundary(x, on_boundary):
    return on_boundary

bcs = [DirichletBC(U, Constant((0.0, 0.0, 0.0)), all_boundary)]

# Now assemble reduced terms using our custom assembler
a_01 = Form(a_01)
a_10 = Form(a_10)
a_11 = Form(a_11)

import fenics_shells
A = PETScMatrix()
assemble(a_00, tensor=A, finalize_tensor=False)
fenics_shells.MITCAssembler().assemble(A, U, a_01, a_10, a_11)
# Before finalising the matrix
A.apply("add")

# Assemble lhs
L = f*z_t*dx
b = PETScVector()
assemble(L, tensor=b)

for bc in bcs:
    bc.apply(A, b)

u_h = Function(U)
solver = PETScKrylovSolver()
solver.solve(A, u_h.vector(), b)

r_h, z_h = u_h.split()
File("solution.pvd") << z_h 

z_e_h = project(z_e, FunctionSpace(mesh, "CG", 1))

print('------ t = %g    ------' % float(tv))
print('Solution with our projected MITC : %g' %z_h(0.5, 0.5))
print('Lovadina analytical solution     : %g' %z_e_h(0.5, 0.5))

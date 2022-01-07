from dolfin import *
from fenics_shells import *

mesh = UnitSquareMesh(64, 64)

element = MixedElement([VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1),
                        FiniteElement("N1curl", triangle, 1),
                        FiniteElement("N1curl", triangle, 1)])

U = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
U_F = U.full_space

u_ = Function(U_F)
theta_, w_, R_gamma_, p_ = split(u_)
u = TrialFunction(U_F)
u_t = TestFunction(U_F)

E = Constant(10920.0)
nu = Constant(0.3)
kappa = Constant(5.0/6.0)
t = Constant(0.0001)

k = sym(grad(theta_))

D = (E*t**3)/(24.0*(1.0 - nu**2))
psi_M = D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)

psi_T = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(R_gamma_, R_gamma_)

f = Constant(1.0)
W_ext = inner(f*t**3, w_)*dx

gamma = grad(w_) - theta_


L_R = inner_e(gamma - R_gamma_, p_)
L = psi_M*dx + psi_T*dx + L_R - W_ext

F = derivative(L, u_, u_t)
J = derivative(F, u_, u)

A, b = assemble(U, J, -F)


def all_boundary(x, on_boundary):
    return on_boundary

def left(x, on_boundary):
    return on_boundary and near(x[0], 0.0)

def right(x, on_boundary):
    return on_boundary and near(x[0], 1.0)

def bottom(x, on_boundary):
    return on_boundary and near(x[1], 0.0)

def top(x, on_boundary):
    return on_boundary and near(x[1], 1.0)

# Simply supported boundary conditions.
bcs = [DirichletBC(U.sub(1), Constant(0.0), all_boundary),
       DirichletBC(U.sub(0).sub(0), Constant(0.0), top),
       DirichletBC(U.sub(0).sub(0), Constant(0.0), bottom),
       DirichletBC(U.sub(0).sub(1), Constant(0.0), left),
       DirichletBC(U.sub(0).sub(1), Constant(0.0), right)]

for bc in bcs:
    bc.apply(A, b)

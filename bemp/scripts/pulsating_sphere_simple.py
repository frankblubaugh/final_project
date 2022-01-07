import bempp.api
import numpy as np
import matplotlib.pyplot as plt

ka = np.arange(0.1, 10.1, 0.1)
a = 1
r = 1000
c = 1500
rho = 1000
p_field = np.zeros(ka.shape, dtype='complex')
for ii in np.arange(0, ka.size):
    print(ka[ii])
    k = ka[ii]/a
    omega = c*k
    grid = bempp.api.shapes.sphere(h=0.1)
    piecewise_const_space = bempp.api.function_space(grid, "DP", 0)
    @bempp.api.complex_callable
    def f(x, n, domain_index, result):
        result[0] = 1

    vn = bempp.api.GridFunction(piecewise_const_space, fun=f)
    Id = bempp.api.operators.boundary.sparse.identity(
        piecewise_const_space,
        piecewise_const_space,
        piecewise_const_space)
    Mk = bempp.api.operators.boundary.helmholtz.double_layer(
        piecewise_const_space,
        piecewise_const_space,
        piecewise_const_space,
        k)
    Lk = bempp.api.operators.boundary.helmholtz.single_layer(
        piecewise_const_space,
        piecewise_const_space,
        piecewise_const_space,
        k)
    lhs = Mk-0.5*Id
    rhs = Lk*vn
    phi, info = bempp.api.linalg.gmres(lhs, rhs, tol=1e-6)
    p = 1j*rho*omega*phi
    points = np.vstack([0, 0, r])
    Mk_pot = bempp.api.operators.potential.helmholtz.double_layer(
        piecewise_const_space,
        points, k)
    Lk_pot = bempp.api.operators.potential.helmholtz.single_layer(
        piecewise_const_space,
        points, k)
    phi_field = Mk_pot*phi-Lk_pot*vn
    p_field[ii] = 1j*rho*omega*phi_field

plt.figure(dpi=300)
plt.plot(ka, 20*np.log10(np.abs(p_field/1e-6)))
plt.xlabel('ka')
plt.ylabel('SPL')
plt.grid()
pulsating_sphere_data = {'ka': ka, 'p_field': p_field}
np.save('pulsating_sphere_bempp.npy', pulsating_sphere_data)

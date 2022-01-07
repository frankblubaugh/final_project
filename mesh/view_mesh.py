import pyvista
pyvista.set_plot_theme("document")

p = pyvista.Plotter(window_size=(800, 800))
p.add_mesh(
    mesh=pyvista.read("extrude_box.msh"),
    scalars="gmsh:physical",
    stitle="Materials",
    show_scalar_bar=True,
    show_edges=True,
)
p.view_xy()
p.show()

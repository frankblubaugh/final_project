import gmsh

# determine max characteristic length
n_elements_per = 10
c = 340
f_max = 200.
max_length_lambda = c/f_max/n_elements_per/10
min_length_lambda = max_length_lambda/10
print(f'Wavlength for {f_max} Hz is {max_length_lambda*n_elements_per:.2e} m')
print(f'CharacteristicLengthMin: {min_length_lambda:.2e} CharacteristicLengthMax{max_length_lambda:.2e}')


gmsh.initialize()
plate = gmsh.model.occ.add_box(0,0,0,1,1,0.0127)
gmsh.model.occ.synchronize()
gdim = 3
status = gmsh.model.addPhysicalGroup(gdim, [plate], 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin",min_length_lambda)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",max_length_lambda)
gmsh.model.mesh.generate(gdim)

gmsh.write(f"extruded_box_panel_{int(f_max)}.msh")

from dolfin import *     
import meshio

msh = meshio.read(f"extruded_box_panel_{int(f_max)}.msh")
meshio.write(f"extruded_box_panel_{int(f_max)}.xml",msh)

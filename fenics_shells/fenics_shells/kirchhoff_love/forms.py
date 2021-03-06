# -*- coding: utf-8 -*-

# Copyright (C) 2015 Jack S. Hale
#
# This file is part of fenics-shells.
#
# fenics-shells is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# fenics-shells is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with fenics-shells. If not, see <http://www.gnu.org/licenses/>.

from dolfin import *
from fenics_shells import k


def theta(w):
    r"""Returns the rotations as a function of the transverse displacements
    according to the Kirchoff-Love kinematics.

      ..math::
      \theta = \nabla w

    Args:
        w: Transverse displacement field, typically a UFL scalar or a DOLFIN Function

    Returns:
        the rotations with shape (2,)
    """
    return grad(w)

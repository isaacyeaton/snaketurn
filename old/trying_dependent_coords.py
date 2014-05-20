from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import pydy
import sympy as sym
import sympy.physics.mechanics as me

from scipy.linalg import solve, lstsq
from scipy.integrate import odeint

import time
import functools

me.Vector.simp = False  # to increase computation speed

from sympy.physics.vector import init_vprinting
init_vprinting()

from matplotlib import animation
from matplotlib.patches import Ellipse, Rectangle

from IPython.html import widgets
from IPython.html.widgets import interact, interactive

# my own imports
import nchain


n = 3

dynamic, symbol, setup, fbd = nchain.formulate_nchain_parameters(n)
eom = nchain.make_kane_eom(dynamic, setup, fbd)


q = np.array(dynamic['q'])
u = np.array(dynamic['u'])

#kd = np.array(fbd['kd'])[[0, 3, 4]].tolist()

q_ind = q[[0, 3, 4]].tolist()
u_ind = u[[0, 3, 4]].tolist()

q_dep = q[[1, 2]].tolist()
u_dep = u[[1, 2]].tolist()

q_cons = [q[1] - q[0], q[2] - q[1]]
u_cons = [u[1] - u[0], u[2] - u[1]]

#q_cons = [0, 0]
#u_cons = [0, 0]


# equations of motion using Kane's method
kane = me.KanesMethod(frame=setup['frames'][0],
                      q_ind=q_ind,
                      u_ind=u_ind,
                      kd_eqs=fbd['kd'],
                      q_dependent=q_dep,
                      configuration_constraints=q_cons,
                      u_dependent=u_dep,
                      velocity_constraints=u_cons)
(fr, frstar) = kane.kanes_equations(fbd['fl'], fbd['bodies'])
kanezero = fr + frstar
mass = kane.mass_matrix_full
forcing = kane.forcing_full

eom = dict(kane=kane, fr=fr, frstar=frstar, mass=mass, forcing=forcing, kanzero=kanezero)

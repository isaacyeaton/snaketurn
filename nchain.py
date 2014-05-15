from __future__ import division

import numpy as np
import sympy as sym
import sympy.physics.mechanics as me
me.Vector.simp = False  # to increase computation speed


from pydy_code_gen.code import generate_ode_function


def formulate_nchain_parameters(n):
    """Formulate the n-chain system, returning a bunch of sympy objects
    to make the equations of motion.

    Parameters
    ----------
    n : int
        Number of links

    Returns
    -------
    q : list
        Generalized coordinates
    u : list
        Generalized speeds
    T : list
        Torques
    m : list
        Masses of each segment
    l : list
        Lengths of each segment
    Izz : list
        Moments of inertia about center of mass
    """

    # dynamic variables
    q = me.dynamicsymbols('q:' + str(n + 2))
    u = me.dynamicsymbols('u:' + str(n + 2))
    T = me.dynamicsymbols('T:' + str(n + 1))

    # variables for bodies
    g, t = sym.symbols('g t')
    m = sym.symbols('m:' + str(n))
    l = sym.symbols('l:' + str(n))
    Izz = sym.symbols('I_zz_:' + str(n))

    # frames
    frames = [me.ReferenceFrame('fr_N')]
    for i in range(n):
        name = 'fr_' + str(i)
        frame = frames[i].orientnew(name, 'Axis', [q[i], frames[i].z])
        frame.set_ang_vel(frames[i], u[i] * frames[i].z)
        frames.append(frame)

    # origin of inertial frame
    joints = [me.Point('N_or')]
    joints[0].set_vel(frames[0], 0)

    # joints
    for i in range(n + 1):
        name = 'jt_' + str(i)
        if i == 0:
            joint = joints[i].locatenew(name, q[-2] * frames[0].x + q[-1] * frames[0].y)
            joint.set_vel(frames[0], u[-2] * frames[0].x + u[-1] * frames[0].y)
        else:
            joint = joints[i].locatenew(name, l[i - 1] * frames[i].x)
            joint.v2pt_theory(joints[i], frames[0], frames[i])
        joints.append(joint)

    # mass centers
    mc = []
    for i in range(n):
        name = 'mc_' + str(i)
        mcent = joints[i + 1].locatenew(name, l[i] / 2 * frames[i + 1].x)
        mcent.v2pt_theory(joints[i + 1], frames[0], frames[i + 1])
        mc.append(mcent)

    # inertia dyads
    dyads = []
    for i in range(n):
        dyad = (me.inertia(frames[i + 1], 0, 0, Izz[i]), mc[i])
        dyads.append(dyad)

    # bodies
    bodies = []
    for i in range(n):
        name = 'bd_' + str(i)
        body = me.RigidBody(name, mc[i], frames[i + 1], m[i], dyads[i])
        bodies.append(body)

    # kinematic differential equations
    kd = []
    for i in range(n + 2):
        kd.append(q[i].diff(t) - u[i])

    forces = []
    for i in range(n):
        forces.append((mc[i], -m[i] * g * frames[0].z))

    torques = []
    for i in range(n):
        torques.append((frames[i + 1], (T[i] - T[i + 1]) * frames[0].z))

    fl = forces + torques

    # setup dictionaries of values to return
    dynamic = dict(q=q, u=u, T=T)
    symbol = dict(m=m, l=l, Izz=Izz, g=g, t=t)
    setup = dict(frames=frames, joints=joints, mc=mc, dyads=dyads)
    fbd = dict(kd=kd, fl=fl, bodies=bodies)

    return dynamic, symbol, setup, fbd


def make_kane_eom(dynamic, setup, fbd):
    """Formulate the equations of motion using Kane's method.

    Paramters
    ---------
    The dictionaries that are returned from formulated_nchain_parameters

    Returns
    -------

    """

    # equations of motion using Kane's method
    kane = me.KanesMethod(frame=setup['frames'][0],
                          q_ind=dynamic['q'],
                          u_ind=dynamic['u'],
                          kd_eqs=fbd['kd'],
                          q_dependent=[],
                          configuration_constraints=[],
                          u_dependent=[],
                          velocity_constraints=[])
    (fr, frstar) = kane.kanes_equations(fbd['fl'], fbd['bodies'])
    mass = kane.mass_matrix_full
    forcing = kane.forcing_full

    eom = dict(kane=kane, fr=fr, frstar=frstar, mass=mass, forcing=forcing)

    return eom


def make_lagrange_eom(dynamic, setup, fbd):
    """Formulate the equations of motion using Lagranges's method.

    Paramters
    ---------
    The dictionaries that are returned from formulated_nchain_parameters

    Returns
    -------

    """

    # calculate the Lagrangian
    L = me.Lagrangian(setup['frames'][0], *fbd['bodies'])
    lagrange = me.LagrangesMethod(L, dynamic['q'], forcelist=fbd['fl'], frame=setup['frames'][0])

    langeqn = lagrange.form_lagranges_equations()
    mass = lagrange.mass_matrix_full
    forcing = lagrange.forcing_full

    eom = dict(L=L, lagrange=lagrange, langeqn=langeqn, mass=mass, forcing=forcing)

    return eom

from __future__ import division

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse, Rectangle

import sympy as sym
import sympy.physics.mechanics as me
me.Vector.simp = False  # to increase computation speed

from pydy_code_gen.code import generate_ode_function

import functools


def printer(exp):
    """Print the expression so it can be copied into latex.
    """
    return str(me.mlatex(exp))

def formulate_nchain_parameters(n):
    """Formulate the n-chain system, returning a bunch of sympy objects
    to make the equations of motion.

    Parameters
    ----------
    n : int
        Number of links

    Returns
    -------
    TODO: fixme These are acutally 4 dicts filled with lists
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

    fl = forces #+ torques

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
    kanezero = fr + frstar
    mass = kane.mass_matrix_full
    forcing = kane.forcing_full

    eom = dict(kane=kane, fr=fr, frstar=frstar, mass=mass, forcing=forcing, kanzero=kanezero)

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


def equal_snake(n, mtot=40.5, ltot=68.6, wid=2.2):
    """Get masses, lengths, and moments of inertia for snake, assuming
    each segment is the same.

    Parameters
    ----------
    n : int
        Number of segments
    mtot : float
        Total mass of the snake in *grams*
    ltot : float
        Total length of the snake in *cm*
    wid : float (default=2)
        Width of the snake in *cm*

    Returns
    -------
    masses, lengths, widths, mom_inertia : lists
        Equal snake parameters

    Note
    ----
    The moment of inertias are calculated based on a filled half cylinder
    with radius of `width'. This approximate will become less valid as
    `n' increases and the aspect ratio of the snake decreases.
    """

    # normalized each chain
    norm = np.ones(n) / n

    masses = 40.5 / 1000 * norm  # kg
    lengths = 68.6 / 100 * norm  # m
    widths = wid / 100 * norm  # m

    #mom_inertias = masses * (lengths**2 + width**2) / 12
    mom_inertias = masses / 4 * widths**2 + masses / 12 * lengths**2

    # store the values in a dictionary
    snake = dict(m=masses, l=lengths, w=widths, I=mom_inertias)

    return snake


def callable_matrices(dynamic, eom, consts):
    """Take the equations of motion and make the mass matrix
    and forcing vector callable so we can simulate the system.

    Parameters
    ----------

    Returns
    -------


    """

    # dummy symbols used for dynamamic variables
    dyn_symbols = dynamic['q'] + dynamic['u']
    dummy_symbols = [sym.Dummy() for i in dyn_symbols]
    dummy_dict = dict(zip(dyn_symbols, dummy_symbols))

    # substitute dummy symbols into mass matrix and forcing vector
    kd_dict = eom['kane'].kindiffdict()                              # solved kinematical differential equations
    F = eom['forcing'].subs(kd_dict).subs(dummy_dict)      # substitute into the forcing vector
    M = eom['mass'].subs(kd_dict).subs(dummy_dict)  # substitute into the mass matrix

    # callable matrices
    F_func = sym.lambdify(consts['names'] + dummy_symbols, F)
    M_func = sym.lambdify(consts['names'] + dummy_symbols, M)

    # partially evaluate matrices to run faster (maybe 15% or so)
    ffunc = functools.partial(F_func, *consts['vals'])
    mfunc = functools.partial(M_func, *consts['vals'])

    return F_func, M_func, ffunc, mfunc


def organize_data(y, n, snake_dict):
    """Organize the data so it can be animated with ellipses.
    """

    ms = snake_dict['m']
    ls = snake_dict['l']
    npts = y.shape[0]
    angs = y[:, :n]
    rads = angs.cumsum(axis=1)
    degs = np.rad2deg(rads)

    jtx, jty = y[:, n:n + 2].T
    jtx = jtx.reshape(-1, 1)
    jty = jty.reshape(-1, 1)

    mcx, mcy = [], []
    for i in range(n):
        dx = ls[i] * np.cos(rads[:, i])
        dy = ls[i] * np.sin(rads[:, i])

        mcx.append((jtx[:, i] + dx / 2).flatten())
        mcy.append((jty[:, i] + dy / 2).flatten())

        x = (jtx[:, i] + dx).reshape(-1, 1)
        y = (jty[:, i] + dy).reshape(-1, 1)

        jtx = np.hstack((jtx, x))
        jty = np.hstack((jty, y))

    mcx = np.array(mcx).T
    mcy = np.array(mcy).T

    # center of mass
    mms = np.array(ms)
    comx = (mcx * mms).sum(axis=1) / mms.sum()
    comy = (mcy * mms).sum(axis=1) / mms.sum()

    return rads, degs, jtx, jty, mcx, mcy, comx, comy


def n_movie_maker(n, snake_dict, ts, y, limx=(-1, 1), limy=(-1, 1), hst=50, interval=100):

    rads, degs, jtx, jty, mcx, mcy, comx, comy = organize_data(y, n, snake_dict)

    ms = snake_dict['m']
    ls = snake_dict['l']
    ws = snake_dict['w']
    ii = 0
    st = 0

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9, 9))
    ax.axis('image')
    ax.set_xlim(limx)
    ax.set_ylim(limy)

    fig.set_facecolor('w')
    fig.tight_layout()

    time_fmt = '{0:2.3} sec'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize='large')

    ellipse_params = dict(alpha=.5, color='green')
    patches = []
    for i in range(n):
        e = Ellipse(xy=(mcx[ii, i], mcy[ii, i]),
                    width=ws[i], height=1.05 * ls[i],
                    angle=degs[ii, i] - 90, **ellipse_params)
        ax.add_patch(e)
        patches.append(e)

    head, = ax.plot(jtx[ii, 0], jty[ii, 0], 'go', alpha=.6, markersize=7)
    jlines, mclines = [], []
    for i in range(n + 1):
        jline, = ax.plot(jtx[st:ii, i], jty[st:ii, i], 'b--', alpha=.5)
        jlines.append(jline)
    for i in range(n):
        mcline, = ax.plot(mcx[st:ii, i], mcy[st:ii, i], 'r', alpha=.5)
        mclines.append(mcline)

    coml, = ax.plot(comx[st:ii], comy[st:ii], 'k-', lw=2, alpha=.9)

    def init():
        ii = 0
        head.set_data(jtx[ii, 0], jty[ii, 0])
        for i in range(n):
            patches[i].set_alpha(.3)
        time_text.set_text('')
        return [head, time_text] + patches

    def animate(ii):

        st = ii - hst
        if st < 0:
            st = 0

        head.set_data(jtx[ii, 0], jty[ii, 0])
        for i in range(n):
            patches[i].set_visible(True)
            patches[i].set_alpha(.5)
            patches[i].center = (mcx[ii, i], mcy[ii, i])
            patches[i].angle = degs[ii, i] - 90
            jlines[i].set_data(jtx[st:ii, i], jty[st:ii, i])
            mclines[i].set_data(mcx[st:ii, i], mcy[st:ii, i])
        for i in range(n + 1):
            jlines[i].set_data(jtx[st:ii, i], jty[st:ii, i])

        time_text.set_text(time_fmt.format(ts[ii]))

        coml.set_data(comx[st:ii], comy[st:ii])

        return patches + [head, coml, time_text] + jlines + mclines

    anim = animation.FuncAnimation(fig, animate, frames=len(ts),
                interval=interval, blit=True, repeat=False, init_func=init)

    return anim
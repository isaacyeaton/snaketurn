from __future__ import division

import numpy as np
import importlib

def load_cython(filename_prefix, ncoords, specified=True):
    """Load in precompiled cython code.

    Parameters
    ----------
    filename_prefix : str
        Name of cython module to load (without extension)
    ncoords : int
        Number of generalized coordinates
    specified : bool, default=True
        If there are time varying parameters in the simulation.

    Returns
    -------
    rhs : function
        Function that gives the derivates of the state variables

    See https://github.com/pydy/pydy/blob/master/pydy/codegen/code.py#L408
    """


    cython_module = importlib.import_module(filename_prefix)
    mass_forcing_func = cython_module.mass_forcing_matrices

    def eval_ode(x, t, args):
        print args
        segmented = [args['constants'],
                    x[:ncoords],
                    x[ncoords:]]

        if specified is not None:
            try:
                sp_val = args['specified'](x, t)
            except TypeError:  # not callable
                # If not callable, then it should be a float or ndarray.
                sp_val = args['specified']

            # If the value is just a float, then convert to a 1D array.
            try:
                len(sp_val)
            except TypeError:
                sp_val = np.asarray([sp_val])

            segmented.append(sp_val)

        mass_matrix_values, forcing_vector_values = \
            mass_forcing_func(*segmented)

        # TODO: figure out how to off load solve to the various generated
        # code, for example for Theano:
        # http://deeplearning.net/software/theano/library/sandbox/linalg.html#theano.sandbox.linalg.ops.Solve

        # Could use scipy.linalg.solve and enable a and b overwriting to
        # avoid the array copying.
        dx = np.array(np.linalg.solve(mass_matrix_values,
                                      forcing_vector_values)).T[0]

        return dx

    return eval_ode
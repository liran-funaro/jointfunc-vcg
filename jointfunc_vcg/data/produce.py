"""
Author: Liran Funaro <liran.funaro@gmail.com>

Copyright (C) 2006-2018 Liran Funaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import functools
import vecfunc
from scipy.interpolate import CubicSpline


def get_val_fake_x(sd, shape, ndim=None):
    if type(shape) in (list, tuple):
        val_fake_x = [np.linspace(0, 1, s) for s in shape]
    else:
        assert ndim <= sd.ndim
        val_fake_x = [np.linspace(0, 1, shape) for _ in range(ndim)]
    return val_fake_x


def get_val_x(sd, shape, ndim=None):
    if type(shape) in (list, tuple):
        val_x = [np.arange(0, s) for s in shape]
    else:
        assert ndim <= sd.ndim
        val_x = [np.arange(0, shape) for _ in range(ndim)]
    return val_x


def get_vals_spline(sd):
    val_spline = sd.internal.get('val-spline', None)
    if val_spline is None:
        val_xy = sd.data['val-xy']
        val_spline = [[CubicSpline(np.array(v[0]), np.array(v[1]), bc_type='natural') for v in vs] for vs in val_xy]
    return val_spline


def get_vals_slices(sd, shape, ndim=None, factor_wealth=True):
    val_spline = get_vals_spline(sd)
    val_fake_x = get_val_fake_x(sd, shape, ndim)

    ret = [[v(x) for v, x in zip(vs, val_fake_x)] for vs in val_spline]
    if factor_wealth:
        wealth = sd.dist_data['wealth']
        for i in range(sd.n):
            for d in range(ndim):
                ret[i][d] *= wealth[i]

    is_concave = sd.meta['valuation'].setdefault('concave', False)
    r = sd.meta['valuation'].setdefault('local-maximum-limit', None)
    is_rising = r is None or r <= 0

    for i, vs in enumerate(ret):
        for d, v in enumerate(vs):
            if is_concave:
                ret[i][d] = vecfunc.fix_concave_rising(v).arr
            elif is_rising:
                ret[i][d] = vecfunc.fix_rising(v).arr
    return ret


def get_vals(sd, shape, ndim, factor_wealth=True, players=None, resource_dependency=None):
    if isinstance(shape, int):
        shape = (shape,)
    if not type(shape) in (list, tuple):
        raise TypeError("Shape must be a tuple or an int.")
    assert ndim == len(shape)

    if players is None:
        players = range(sd.n)

    val_slices = get_vals_slices(sd, shape, ndim, factor_wealth=False)
    val_slices = [val_slices[p] for p in players]
    meshes = (np.meshgrid(*vs, sparse=False, indexing='ij') for vs in val_slices)

    if resource_dependency is None:
        resource_dependency = 'multiply'
    resource_dependency_func_options = {
        'complementary': np.minimum,
        'c': np.minimum,
        'substitute': np.maximum,
        's': np.maximum,
        'multiply': np.multiply,
        'm': np.multiply,
    }

    resource_dependency_func = resource_dependency_func_options.get(resource_dependency.lower())

    if resource_dependency_func:
        vals = [functools.reduce(resource_dependency_func, m) for m in meshes]
    else:
        k = f'resource_dependency_{resource_dependency}'
        if k not in sd.data:
            raise KeyError(f"No such resource dependency: {resource_dependency}.")
        dep = sd.data[k]
        dep = [dep[p] for p in players]

        vals = []
        for m, d in zip(meshes, dep):
            mm = {i: sm for i, sm in enumerate(m)}
            assert len(mm) == ndim, f'Initial len: {len(mm)}'
            for a, (i0, i1) in d:
                n0 = mm.pop(i0, None)
                n1 = mm.pop(i1, None)
                if n0 is None:
                    mm[i1] = n1
                elif n1 is None:
                    mm[i1] = n0
                else:
                    resource_dependency_func = resource_dependency_func_options[a.lower()]
                    mm[i1] = resource_dependency_func(n0, n1)

            assert len(mm) == 1, f'Result len: {len(mm)}'
            vals.append(mm[next(iter(mm))])

    if factor_wealth:
        wealth = sd.dist_data['wealth']
        for i, p in enumerate(players):
            vals[i] *= wealth[p]

        if ndim > 1:
            part_wealth = np.divide(wealth, ndim)
            for i, p in enumerate(players):
                for d in range(ndim):
                    val_slices[i][d] *= part_wealth[p]

    return val_slices, vals

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
from jointfunc_vcg.data import produce
from jointfunc_vcg.exp import param
import vecfunc_vcg
from vecfunc_vcg.vecfuncvcglib import joint_func

import numpy as np


#########################################################################################################
# Maille Tuffin (1D Concave Comparison)
#########################################################################################################
def maille_tuffin(_ds_obj, index, sd, sz, ndim, resource_dependency):
    shape = param.get_shape_for_gridpoints(sz, ndim)
    gridpoints = np.prod(shape)
    n_chunks = np.subtract(shape, 1)
    val_slices, vals = produce.get_vals(sd, shape, ndim, factor_wealth=True, resource_dependency=resource_dependency)
    ret = vecfunc_vcg.maille_tuffin(vals, val_slices, n_chunks)
    return {
        'input': {
            'index': index,
            'sz': sz,
            'gridpoints': gridpoints,
            'shape': shape,
            'n-chunks': n_chunks,
            'ndim': ndim,
        },
        **ret
    }


#########################################################################################################
# Joint Valuation
#########################################################################################################
def joint_val(_ds_obj, index, sd, sz, ndim, join_method=3, join_chunk_size=8, join_flags=None,
              resource_dependency='complementary'):
    shape = param.get_shape_for_gridpoints(sz, ndim)
    gridpoints = np.prod(shape)
    n_chunks = np.subtract(shape, 1)
    val_slices, vals = produce.get_vals(sd, shape, ndim, factor_wealth=True, resource_dependency=resource_dependency)
    ret = vecfunc_vcg.joint_func(vals, n_chunks, join_method=join_method, join_chunk_size=join_chunk_size,
                                 join_flags=join_flags)
    return {
        'input': {
            'index': index,
            'sz': sz,
            'gridpoints': gridpoints,
            'shape': shape,
            'n-chunks': n_chunks,
            'ndim': ndim,
            'join-method': join_method,
            'join-chunk-size': join_chunk_size,
        },
        **ret
    }


#########################################################################################################
# Test data structure build time
#########################################################################################################
def test_joint_val_ds_build_time(_ds_obj, index, sd, sz, ndim, join_method=3, join_chunk_size=128,
                                 resource_dependency='complementary'):
    shape = param.get_shape_for_gridpoints(sz, ndim)
    gridpoints = np.prod(shape)
    n_chunks = np.subtract(shape, 1)
    val_slices, vals = produce.get_vals(sd, shape, ndim, factor_wealth=True, resource_dependency=resource_dependency)
    ret = joint_func.sum_test_ds_build_time(vals, method=join_method, chunk_size=join_chunk_size)
    return {
        'input': {
            'index': index,
            'sz': sz,
            'gridpoints': gridpoints,
            'shape': shape,
            'n-chunks': n_chunks,
            'ndim': ndim,
            'join-method': join_method,
            'join-chunk-size': join_chunk_size,
        },
        'stats': ret,
    }

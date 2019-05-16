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
import os
import numpy as np

from cloudsim import dataset


def get_experiment_name(exp_type, exp_param=None, exp_prefix=None, exp_suffix=None):
    if type(exp_param) in (tuple, list):
        exp_param = "-".join(map(str, exp_param))

    key_set = []
    for key in (exp_prefix, exp_type, exp_param, exp_suffix):
        if key is None:
            continue

        if not isinstance(key, slice):
            key = dataset.convert_to_nice_filename(key)
            key = key.replace(os.path.sep, ":")
        key_set.append(key)

    return tuple(key_set)


def get_shape_for_gridpoints(sz, ndim):
    """ Finds a multidimensional, balanced, shape that have the closest number of gridpoints """
    t = int(float(sz)**(1/ndim))
    ret = [t] * (ndim-1)
    total = np.prod(ret)
    ret.append(int(np.round(sz / total)))
    return tuple(ret)

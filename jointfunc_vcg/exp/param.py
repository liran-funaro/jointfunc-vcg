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


def get_experiment_name(exp_type, exp_param=None, exp_prefix=None):
    if type(exp_param) in (tuple, list):
        exp_param = "-".join(map(str, exp_param))

    if exp_prefix is not None:
        exp_type = "%s-%s" % (exp_prefix, exp_type)

    ret = []
    for val in (exp_type, exp_param):
        if val is None:
            continue

        val = dataset.convert_to_nice_filename(val)
        val = val.replace(os.path.sep, ":")
        ret.append(val)

    return tuple(ret)


def get_shape_for_gridpoints(sz, ndim):
    """ Finds a multidimensional, balanced, shape that have the closest number of gridpoints """
    t = int(float(sz)**(1/ndim))
    ret = [t] * (ndim-1)
    total = np.prod(ret)
    ret.append(int(np.round(sz / total)))
    return tuple(ret)

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
from cloudsim.dataset import DataSet
from jointfunc_vcg.exp import batch, workers, param


#########################################################################################################
# Generic Experiments Function
#########################################################################################################

def start(ds_obj: DataSet, worker_func, exp_type, exp_param=None, exp_prefix=None, exp_suffix=None,
          sim_kwargs=None, **kwargs):
    if sim_kwargs is None:
        sim_kwargs = {}
    sim_key = param.get_experiment_name(exp_type, exp_param, exp_prefix, exp_suffix)
    ret = ds_obj.create_job(sim_key, worker_func, kwargs=kwargs, **sim_kwargs)
    ds_obj.clear_cache()
    return ret


#########################################################################################################
# Specific Experiments Functions
#########################################################################################################

def maille_tuffin(ds_obj: DataSet, exp_type='maille-tuffin', ndim=1, sz=2 ** 10,
                  resource_dependency='complementary',
                  exp_prefix=None, sim_kwargs=None, **kwargs):
    return start(ds_obj, workers.maille_tuffin,
                 exp_type=exp_type,
                 exp_param=(ndim, sz),
                 exp_prefix=exp_prefix,
                 exp_suffix=resource_dependency,
                 sim_kwargs=sim_kwargs,
                 sz=sz, ndim=ndim, resource_dependency=resource_dependency, **kwargs)


def joint_val(ds_obj: DataSet, exp_type='joint-val', join_method=3, ndim=1, sz=2 ** 10,
              resource_dependency='complementary',
              exp_prefix=None, sim_kwargs=None, **kwargs):
    return start(ds_obj, workers.joint_val,
                 exp_type=exp_type,
                 exp_param=(join_method, ndim, sz),
                 exp_prefix=exp_prefix,
                 exp_suffix=resource_dependency,
                 sim_kwargs=sim_kwargs,
                 join_method=join_method, sz=sz, ndim=ndim, resource_dependency=resource_dependency, **kwargs)


def test_joint_val_ds_build_time(ds_obj: DataSet, exp_type='test-buildtime', join_method=3, sz=2 ** 10, ndim=1,
                                 resource_dependency='complementary',
                                 exp_prefix=None, sim_kwargs=None, **kwargs):
    return start(ds_obj, workers.test_joint_val_ds_build_time,
                 exp_type=exp_type,
                 exp_param=(join_method, ndim, sz),
                 exp_prefix=exp_prefix,
                 exp_suffix=resource_dependency,
                 sim_kwargs=sim_kwargs,
                 join_method=join_method, sz=sz, ndim=ndim, resource_dependency=resource_dependency, **kwargs)

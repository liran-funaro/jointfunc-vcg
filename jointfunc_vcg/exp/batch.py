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
from cloudsim import dataset, job
from jointfunc_vcg import exp, data
import numpy as np
import pandas as pd


def maille_tuffin(ds_obj: dataset.DataSet, exp_type='maille-tuffin', sizes=(2**10,), dims=(1,),
                  resource_dependencies=('complementary', 'substitute', 'multiply'), **kwargs):
    for d in dims:
        for sz in sizes:
            for r in resource_dependencies:
                job.add_batch_job(exp.maille_tuffin, ds_obj,
                                  exp_type=exp_type,
                                  sz=sz, ndim=d, resource_dependency=r, **kwargs)


def joint_val(ds_obj: dataset.DataSet, exp_type='joint-val', join_methods=(0,), sizes=(2**10,), dims=(1,),
              resource_dependencies=('complementary', 'substitute', 'multiply'), **kwargs):
    for m in join_methods:
        for d in dims:
            for sz in sizes:
                for r in resource_dependencies:
                    job.add_batch_job(exp.joint_val, ds_obj,
                                      exp_type=exp_type,
                                      join_method=m, sz=sz, ndim=d,
                                      resource_dependency=r, **kwargs)


def ds_build_time(ds_obj: dataset.DataSet, exp_type='test-buildtime',
                  join_methods=(3,), sizes=(2**10,), dims=(1,),
                  resource_dependencies=('complementary', 'substitute', 'multiply'), **kwargs):
    for m in join_methods:
        for d in dims:
            for sz in sizes:
                for r in resource_dependencies:
                    job.add_batch_job(exp.test_joint_val_ds_build_time, ds_obj,
                                      exp_type=exp_type,
                                      join_method=m, sz=sz, ndim=d,
                                      resource_dependency=r, **kwargs)


def get_batch_jobs_list():
    jobs = job.get_batch_jobs_list()

    headers = ['name', 'dataset']
    ret = []
    for job_func, job_args, job_kwargs in jobs:
        line = [None] * len(headers)
        line[0] = job_func.__name__
        line[1] = job_args[0].meta['name']
        for k, v in job_kwargs.items():
            try:
                k_ind = headers.index(k)
                line[k_ind] = v
            except ValueError:
                headers.append(k)
                line.append(v)
        ret.append(line)
    return pd.DataFrame(ret, columns=headers)

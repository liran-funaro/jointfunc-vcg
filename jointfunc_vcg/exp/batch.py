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


def maille_tuffin(ds_obj: dataset.DataSet, exp_type='maille-tuffin', sizes=(2**10,), dims=(1,), **kwargs):
    for d in dims:
        for sz in sizes:
            job.add_batch_job(exp.maille_tuffin, ds_obj,
                              exp_type=exp_type,
                              sz=sz, ndim=d, **kwargs)


def joint_val(ds_obj: dataset.DataSet, exp_type='joint-val', join_methods=(0,), sizes=(2**10,), dims=(1,), **kwargs):
    for m in join_methods:
        for d in dims:
            for sz in sizes:
                job.add_batch_job(exp.joint_val, ds_obj,
                                  exp_type=exp_type,
                                  join_method=m, sz=sz, ndim=d, **kwargs)


def ds_build_time(ds_obj: dataset.DataSet, exp_type='test-buildtime',
                  join_methods=(3,), sizes=(2**10,), dims=(1,), **kwargs):
    for m in join_methods:
        for d in dims:
            for sz in sizes:
                job.add_batch_job(exp.test_joint_val_ds_build_time, ds_obj,
                                  exp_type=exp_type,
                                  join_method=m, sz=sz, ndim=d, **kwargs)


def all_experiments(max_workers=1, datasets_interval=None):
    sim_kwargs = dict(datasets_interval=datasets_interval, max_workers=max_workers)

    main_join_flags = 'filter',
    tested_dims = 1, 2, 3, 4, 5, 6

    # Brute force = O(n * N^2)
    brute_force_sizes = np.linspace(2**8, 2**14, 10).astype(np.uint32)
    for ds_obj in data.concave, data.nonconcave, data.nonrising:
        joint_val(ds_obj, exp_type='brute-force-complexity',
                  join_methods=(0,), sizes=brute_force_sizes, dims=tested_dims,
                  join_flags=main_join_flags,
                  sim_kwargs=sim_kwargs)

    # Maille tuffin va joint val (correct results)
    maille_tuffin_sizes = np.linspace(2**10, 2**16, 20).astype(np.uint32)
    maille_compare_methods = 3, 5
    maille_tuffin(data.concave, exp_type='maille-tuffin', exp_prefix='baseline',
                  sizes=maille_tuffin_sizes, sim_kwargs=sim_kwargs)
    joint_val(data.concave, exp_type='maille-tuffin', exp_prefix='vs-joint-val',
              join_methods=maille_compare_methods, sizes=maille_tuffin_sizes, dims=(1,),
              join_flags=main_join_flags,
              sim_kwargs=sim_kwargs)

    joint_val_sizes = np.linspace(2**10, 2**16, 10).astype(np.uint32)
    compare_methods = 5, 9, 2, 6

    # Data structures compare
    for ds_obj in data.concave, data.nonconcave, data.nonrising:
        joint_val(ds_obj, exp_type='ds-compare',
                  join_methods=compare_methods, sizes=joint_val_sizes, dims=tested_dims,
                  join_flags=main_join_flags,
                  sim_kwargs=sim_kwargs)

    # Data structures analysis
    analysis_join_flags = 'filter', 'count', 'buildtime', 'querytime'
    joint_val(data.nonconcave, exp_type='ds-analysis',
              join_methods=compare_methods, sizes=joint_val_sizes, dims=tested_dims,
              join_flags=analysis_join_flags,
              sim_kwargs=sim_kwargs)

    analysis_compare_methods = 5,
    for ds_obj in data.concave, data.nonrising:
        joint_val(ds_obj, exp_type='ds-analysis',
                  join_methods=analysis_compare_methods, sizes=joint_val_sizes, dims=tested_dims,
                  join_flags=analysis_join_flags,
                  sim_kwargs=sim_kwargs)

    # Compare for different number of maxima points


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

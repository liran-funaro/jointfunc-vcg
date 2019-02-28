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
import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt

from jointfunc_vcg import results


def joint_val_vs_maille_tuffin(ds_obj, exp_type, exp_prefix=None, normalized=False):
    r = results.read_unified_results(ds_obj, exp_type, exp_prefix=exp_prefix)
    x = np.prod(r['input', 'shape'], axis=-1).mean(axis=-1)
    if normalized:
        x /= x[0]

    t1 = r['maille-tuffin', 'stats', 'optimizationRunTime']
    if normalized:
        t1 /= t1[0]
    m1 = np.mean(t1, axis=-1)
    s1 = np.std(t1, axis=-1)
    plt.errorbar(x, m1, yerr=s1, label="Maille Tuffin", marker='s', ms=3, elinewidth=1, linewidth=2, capsize=2)

    t2 = r['joint-val', 'stats', 'optimizationRunTime']
    if normalized:
        t2 /= t2[0]
    m2 = np.mean(t2, axis=-1)
    s2 = np.std(t2, axis=-1)
    plt.errorbar(x, m2, yerr=s2, label="Joint Valuation", marker='o', ms=3, elinewidth=1, linewidth=2, capsize=2)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()


def joint_val_gridpoints(ds_obj, exp_type, exp_prefix=None, val=None,
                         val_exp=None, fit_method=None):
    df = results.analyze.joint_val_gridpoints(ds_obj, exp_type, exp_prefix, val, val_exp)
    headers = list(df)
    val_name = headers[-1]
    # n_colors = len(set(df['#Resources']))
    n_colors = len(set(df['Method']))
    sns.lineplot(x='N Gridpoints', y=val_name, hue='Method', data=df,
                 err_style='bars', style="Resources", legend='full', markers=True,
                 palette=sns.color_palette('hls', n_colors))

    if fit_method is not None:
        groups = [(d, ddf.groupby('N Gridpoints')) for d, ddf in df.groupby(by='Resources')]
        groups_mean = [(d, g, g.mean()) for d, g in groups]
        ret = [(d, list(g.groups.keys()), list(gm[val_name])) for d, g, gm in groups_mean]

        for d, sz, perf in ret:
            x = sz
            y = perf
            try:
                x = np.array(x, dtype=float)
                y = np.array(y, dtype=float)
                xs = np.linspace(1, x[-1], 1024)
                popt, ys, fit = results.analyze.fit_complexity(x, y, xs, fit_method=fit_method)
                print("Resource", d, "fit:", fit)
                plt.plot(xs, ys, color='black', alpha=0.5, linestyle=':')
            except Exception as e:
                print("Could not fit resource", d, "-", e)

    plt.grid(True, linestyle=':', alpha=0.7)
    return df


def joint_val_gridpoints_inverse(ds_obj, exp_type, exp_prefix=None):
    r = results.read_unified_results(ds_obj, exp_type, exp_prefix=exp_prefix)

    ndim = r['input', 'ndim']
    shape = r['input', 'shape']
    perf = r['stats', 'optimizationRunTime']

    df = []
    for cd, cs, cp in zip(ndim, shape, perf):
        df.extend(zip(cd, np.prod(cs, axis=-1), cp))

    df = pd.DataFrame(df, columns=['Resources', 'N Gridpoints', 'Seconds'])

    groups = [(d, ddf.groupby('N Gridpoints')) for d, ddf in df.groupby(by='Resources')]
    groups_mean = [(d, g, g.mean()) for d, g in groups]

    ret = [(d, list(g.groups.keys()), list(gm['Seconds'])) for d, g, gm in groups_mean]
    x_min = max([sz[0] for d, sz, perf in ret])
    x_max = min([sz[-1] for d, sz, perf in ret])
    x = np.linspace(x_min, x_max, 6)

    all_dims = sorted(set(df['Resources']))
    data = np.array([np.interp(x, sz, perf) for d, sz, perf in ret])

    for i, xx in enumerate(x):
        plt.plot(all_dims, data[:, i], label="%.2f" % xx)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    return df

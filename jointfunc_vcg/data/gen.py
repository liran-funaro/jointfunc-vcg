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
import time
import numpy as np

import vecfunc
from cloudsim import stats, azure
from cloudsim.sim_data import SimulationData


def generate_init_data(sd: SimulationData):
    sd.init_seed()
    sd.log("Generating initial data (seed: %d)..." % sd.seed)

    generate_distributions(sd)
    generate_init_valuation(sd)

    return sd


def generate_data(sd: SimulationData):
    if 'generate-time' in sd.meta:
        sd.log("Data already generated.")
        return

    sd.log("Generating data...")

    t1 = time.time()
    generate_valuation(sd)
    generate_time = time.time() - t1

    sd.meta['generate-time'] = generate_time
    sd.log("Generation time: %.2f seconds." % generate_time)
    sd.save()


def generate_distributions(sd: SimulationData):
    """
    Generates distribution:
     - wealth and valuations
    """
    n = sd.n
    ndim = sd.ndim
    dist_data = sd.dist_data
    eps = np.finfo('float32').eps

    sd.log("Generating distributions: ", end="")

    # Valuation frequency
    sd.log("valuation freq", end="")
    val_freq = np.random.randint(3, 8, n*ndim).reshape(n, ndim)
    dist_data['val-freq'] = val_freq

    local_maximum_limit = sd.meta['valuation'].setdefault('local-maximum-limit', None)
    if local_maximum_limit is not None and local_maximum_limit > 0:
        val_local_maximum = np.clip(val_freq-3, 0, local_maximum_limit)
        val_local_maximum = np.array([np.random.randint(0, f + 1) for f in val_local_maximum.flatten()],
                                     dtype=int).reshape(n, ndim)
    else:
        val_local_maximum = np.zeros_like(val_freq, dtype=int)
    dist_data['val-local-maximum'] = val_local_maximum

    # Collect Azure Data
    sd.log(", collect azure data", end="")
    full_azure_data = azure.read_azure_data()
    run_time = np.array(full_azure_data['timestamp vm deleted'] - full_azure_data['timestamp vm created'])
    relevant_players = np.where(run_time > 60)[0]
    azure_players = np.random.choice(relevant_players, n, replace=False)
    dist_data['azure-players'] = azure_players
    all_cores = full_azure_data['vm virtual core count']
    cores = np.array(all_cores)[azure_players]

    # Wealth: Indicate the expected income of the player
    sd.log(", wealth.")

    dist, dist_param = stats.get_scipy_dist(sd.meta['valuation']['wealth-dist'])
    player_min_ppf = dist.cdf(cores*1.1, *dist_param) + eps

    wealth_uniform = np.random.uniform(0 + eps, 1 - eps, n)
    wealth_uniform *= 1-player_min_ppf
    wealth_uniform += player_min_ppf
    dist_data['wealth-uniform'] = wealth_uniform
    dist_data['wealth'] = stats.scipy_dist_ppf(wealth_uniform, sd.meta['valuation']['wealth-dist'])


def generate_init_valuation(sd: SimulationData):
    dist_data = sd.dist_data

    sd.log("Generating valuations: initial func.")

    val_freq = dist_data['val-freq']
    is_concave = sd.meta['valuation'].setdefault('concave', False)
    if is_concave:
        init_val = [[vecfunc.rand.init_sample_1d_concave(0, 1) for _ in fs] for fs in val_freq]
    else:
        val_local_maximum = dist_data['val-local-maximum']
        init_val = [[vecfunc.rand.init_sample_1d_uniform(0, 1, f, True, True, l) for f, l in zip(fs, ls)] for fs, ls
                    in zip(val_freq, val_local_maximum)]

    sd.init_data['val'] = init_val


def generate_valuation(sd: SimulationData):
    sd.log("Generating valuations: refine.")

    init_val = sd.init_data['val']
    val_xy = [[vecfunc.vecinterp.refine_chaikin_corner_cutting_xy(*v) for v in vs] for vs in init_val]
    sd.data['val-xy'] = val_xy


def generate_resource_dependency(sd: SimulationData):
    sd.log("Generating resource dependency.")
    n = sd.n
    ndim = sd.ndim

    # 'c': complementary
    # 's': substitute
    # 'm': multiply
    # output: [('c', 0, 1), ('s', 2,3), ('c', 1,3)]
    dependencies_cs = [], ['c', 's'], [0.7, 0.3]
    dependencies_csm = [], ['c', 's', 'm'], [0.6, 0.3, 0.1]
    dependencies_sm = [], ['s', 'm'], [0.3, 0.7]
    for d, c, p in (dependencies_cs, dependencies_csm, dependencies_sm):
        for _ in range(n):
            t = []
            dim_set = set(range(ndim))
            action_ser = list(np.random.choice(c, p=p, size=ndim-1))
            while len(dim_set) > 1:
                nodes = list(np.random.choice(list(dim_set), size=2, replace=False))
                action = action_ser.pop()
                t.append((action, nodes))
                dim_set.remove(nodes[0])

            d.append(t)

    sd.data['resource_dependency_cs'] = dependencies_cs[0]
    sd.data['resource_dependency_csm'] = dependencies_csm[0]
    sd.data['resource_dependency_sm'] = dependencies_sm[0]

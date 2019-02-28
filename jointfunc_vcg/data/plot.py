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
from matplotlib import pylab as plt
from jointfunc_vcg.data import produce
import vecfunc


#####################################################################
# Wealth
#####################################################################

def plot_wealth(sd):
    dist_data = sd.dist_data
    x = np.arange(sd.n)

    wealth = dist_data['wealth']

    sorted_wealth = np.sort(wealth)
    plt.plot(x, wealth)
    plt.plot(x, sorted_wealth)
    plt.xlabel('Player')
    plt.ylabel('Wealth')
    plt.show()


#####################################################################
# Valuations
#####################################################################

def plot_vals_slice(sd, sz=1024, players=None, range_start=0, range_count=4):
    if players is None:
        players = range(range_start, range_start + range_count)
    else:
        range_count = len(players)

    val_x = produce.get_val_x(sd, sz, sd.ndim)
    val_slices = produce.get_vals_slices(sd, sz, sd.ndim, factor_wealth=True)

    rows = np.ceil(range_count / 4)
    plt.figure(figsize=(16, 4 * rows))
    t = 1
    for i in players:
        plt.subplot(rows, 4, t)
        t += 1
        plt.title('Player: %s' % i)
        for d, (v, x) in enumerate(zip(val_slices[i], val_x)):
            plt.plot(x, v, label=str(d))

        plt.legend()


def plot_val(sd, k, sz=32, ndim=2):
    val_slices, vals = produce.get_vals(sd, sz, ndim, factor_wealth=True, players=(k,))
    vecfunc.plot(vals[0], force_wire_2d=True, linewidth=1)
    plt.title("player: %s" % k)

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
from cloudsim.results import Results, UnifiedResults
from jointfunc_vcg.exp import param
from jointfunc_vcg.results import analyze, plot


#####################################################################################################
# Read Unified Results
#####################################################################################################

def read_results(ds_obj: DataSet, exp_type, exp_param=None, exp_prefix=None, exp_suffix=None):
    """ Returns the raw results data """
    exp_key = param.get_experiment_name(exp_type, exp_param, exp_prefix, exp_suffix)
    return Results(ds_obj, exp_key)


def read_unified_results(ds_obj: DataSet, exp_type,
                         exp_param=slice(None), exp_prefix=None, exp_suffix=None, result_count=0):
    """ Returns the raw unified results data """
    exp_key = param.get_experiment_name(exp_type, exp_param=exp_param, exp_prefix=exp_prefix, exp_suffix=exp_suffix)
    return UnifiedResults(ds_obj, exp_key, result_count)

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
from jointfunc_vcg import results
from scipy.optimize import curve_fit


def verify_joint_val_vs_maille_tuffin(ds_obj, exp_type, exp_prefix=None):
    r = results.read_unified_results(ds_obj, exp_type, exp_prefix=exp_prefix)
    print("Methods:", set([m for m_lst in r['joint-val', 'stats', 'method'] for m in m_lst]))
    print("Methods index:", set([m for m_lst in r['input', 'join-method'] for m in m_lst]))

    a1 = r['maille-tuffin', 'allocations']
    a2 = r['joint-val', 'allocations']
    print("Matching allocations:", np.all(np.squeeze(a1) == np.squeeze(a2)))

    sw1 = r['maille-tuffin', 'sw']
    sw2 = r['joint-val', 'sw']
    print("Matching SW:", np.all(np.isclose(sw1, sw2)))

    p1 = r['maille-tuffin', 'payments']
    p2 = r['joint-val', 'payments']
    print("Matching payments:", np.all(np.isclose(p1, p2)))


def joint_val_data_frame(ds_obj, exp_type, exp_prefix=None, fields=()):
    r = results.read_unified_results(ds_obj, exp_type, exp_prefix=exp_prefix)

    ndim = r['input', 'ndim']
    shape = r['input', 'gridpoints']
    method = r['stats', 'method']
    print("Methods:", set([m for m_lst in method for m in m_lst]))

    result = []
    names = []
    for v in fields:
        val_name = v
        if type(val_name) in (tuple, list):
            val_name = val_name[-1]
        names.append(val_name)
        if v is not None:
            result.append(r[v])

    df = []
    for cd, cs, cm, *cr in zip(ndim, shape, method, *result):
        for ccd, ccs, ccm, *ccr in zip(cd, cs, cm, *cr):
            try:
                df.extend((ccd, ccs, ccm, *cccr) for cccr in zip(*ccr))
            except TypeError:
                df.append((ccd, ccs, ccm, ccr))

    return pd.DataFrame(df, columns=['Resources', 'N Gridpoints', 'Method', *names])


def joint_val_gridpoints(ds_obj, exp_type, exp_prefix=None, val=None, val_exp=None):
    r = results.read_unified_results(ds_obj, exp_type, exp_prefix=exp_prefix)

    ndim = r['input', 'ndim']
    shape = r['input', 'gridpoints']
    method = r['stats', 'method']
    print("Methods:", set([m for m_lst in method for m in m_lst]))

    result = None
    val_name = val
    if type(val_name) in (tuple, list):
        val_name = val_name[-1]
    if val is not None:
        result = r[val]
    elif val_exp is not None:
        assert len(val_exp) > 0
        assert (len(val_exp)-1) % 2 == 0

        all_ops = {
            '+': np.add,
            '-': np.subtract,
            '/': np.divide,
            '*': np.multiply,
            '^': np.power
        }

        result = r[val_exp[0]]
        val_name = val_exp[0]
        if type(val_name) in (tuple, list):
            val_name = val_name[-1]

        for i in range(1, len(val_exp), 2):
            op_name = val_exp[i]
            data_name = val_exp[i+1]
            if type(data_name) in (tuple, list):
                data_name = data_name[-1]
            op = all_ops[val_exp[i]]
            sub_result = r[val_exp[i+1]]
            result = [op(cr, cs) for cr, cs in zip(result, sub_result)]
            val_name += op_name + data_name

    df = []
    for cd, cs, cm, cr in zip(ndim, shape, method, result):
        for ccd, ccs, ccm, ccr in zip(cd, cs, cm, cr):
            try:
                df.extend((ccd, ccs, ccm, cccr) for cccr in ccr)
            except:
                df.append((ccd, ccs, ccm, ccr))

    return pd.DataFrame(df, columns=['Resources', 'N Gridpoints', 'Method', val_name])


def complexity_pow_no_par(x, b):
    return x**b


def complexity_pow_no_par_str(b):
    return "n^(%.2f)" % b


def complexity_pow_2(x, a):
    return a * (x**2)


def complexity_pow_2_str(a):
    return "%g * n^2" % a


def complexity_pow(x, a, b):
    return a * (x**b)


def complexity_pow_str(a, b):
    return "%g * n^(%.2f)" % (a, b)


def complexity_pow_plus_const(x, a, b, c):
    return a * (x**b) + c


def complexity_pow_plus_const_str(a, b, c):
    return "%g * n^(%.2f) + %g" % (a, b, c)


def complexity_pow_logn(x, a, b, c):
    return a * (x**b) * (np.log2(x)**c)


def complexity_pow_logn_str(a, b, c):
    eps = np.finfo(np.float32).eps

    if a < eps:
        return 'const'

    ret = "%g" % a
    if b > eps:
        ret += " * n^(%.2f)" % b
    if c > eps:
        ret += " * log^(%.2f)(n)" % c
    # if d > eps:
    #     ret += " + %g" % d
    return ret


def complexity_pow_plus_nlogn(x, a, b, c, d):
    return a * (x**b) + c * x * np.log2(x) + d


def complexity_pow_plus_nlogn_str(a, b, c, d):
    eps = np.finfo(np.float32).eps

    ret = 'const'

    if a > eps and b > eps:
        ret = "%g * n^(%.2f)" % (a, b)

    if c > eps:
        c_ret = "%g * n * log(n)" % c
        if ret == 'const':
            ret = c_ret
        else:
            ret += ' + ' + c_ret

    if d > eps:
        ret += ' + %g' % d
    return ret


def complexity_pow_2_nlogn(x, a, b, c):
    return (a * (x**2)) + (b * x * (np.log2(x)**c))


def complexity_pow_2_nlogn_str(a, b, c):
    eps = np.finfo(np.float32).eps

    ret = ""
    if a > eps:
        ret = "%g * n^2" % a

    if b > eps:
        b_ret = "%g * n" % b
        if c > eps:
            b_ret += " * log^(%.2f)(n)" % c
        if a > eps:
            ret = "(%s) + (%s)" % (ret, b_ret)
        else:
            ret = b_ret

    return ret


def complexity_nlogn(x, a, b, c):
    return a * x * (np.log2(x)**b) + c


def complexity_nlogn_str(a, b, c):
    return "%g * n * log^(%.2f)(n) + %g" % (a, b, c)


def fit_complexity(x, y, xs, fit_method=None):
    if fit_method is None:
        fit_method = 'complexity_pow_plus_nlogn'
    if isinstance(fit_method, str):
        func = globals()[fit_method]
        str_func = globals()[fit_method+"_str"]
    else:
        func = fit_method
        str_func = lambda *par: str(par)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x_facor = 1
    y_factor = 1
    popt, pcov = curve_fit(func, x / x_facor, y / y_factor, bounds=(0, np.inf), maxfev=2000)
    ys = func(xs/x_facor, *popt) * y_factor
    return popt, ys, str_func(*popt)


from difflib import SequenceMatcher
from heapq import nlargest as _nlargest


# https://stackoverflow.com/questions/50861237/is-there-an-alternative-to-difflib-get-close-matches-that-returns-indexes-l
def get_close_matches_indexes(word, possibilities, n=1, cutoff=0.6):
    """Use SequenceMatcher to return a list of the indexes of the best
    "good enough" matches. word is a sequence for which close matches
    are desired (typically a string).
    possibilities is a list of sequences against which to match word
    (typically a list of strings).
    Optional arg n (default 3) is the maximum number of close matches to
    return.  n must be > 0.
    Optional arg cutoff (default 0.6) is a float in [0, 1].  Possibilities
    that don't score at least that similar to word are ignored.
    """

    if not n > 0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
    result = []
    s = SequenceMatcher()
    s.set_seq2(word)
    for idx, x in enumerate(possibilities):
        s.set_seq1(x)
        if s.real_quick_ratio() >= cutoff and s.quick_ratio() >= cutoff and s.ratio() >= cutoff:
            result.append((s.ratio(), idx))

    # Move the best scorers to head of list
    result = _nlargest(n, result)

    # Strip scores for the best n matches
    return [x for score, x in result]

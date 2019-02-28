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
import math
from cloudsim import dataset
from jointfunc_vcg.data.gen import generate_init_data, generate_data
from jointfunc_vcg.data import gen, plot, produce


r"""
# Amazon
- **t2.nano**: 1 vCPU (actual variable), 0.5 GB (\$0.0058 per Hour)
- **m4.16xlarge**: 64 vCPUS, 256 GB (\$3.2 per Hour)

Source: [aws.amazon.com/ec2/pricing/on-demand/](https://aws.amazon.com/ec2/pricing/on-demand/)

$\Rightarrow \frac{3.2}{0.0058}=551.724\ldots$

Amazon can put $551.724$ **t2.nano** players in a machine of **64 CPUs** and **256 GB**.  
Hence, it have some memory overcommitment and CPU overcommitment (~1:8).  
Most probably, **t2.nano** is more expensive (per unit) and they actually put exatctly 512 **t2.nano** 
players in a single machine, yielding no memory overcommitment and exactly 1:8 CPU overcommitment.

# Valuations Wealth
I used the pareto distribution with $\alpha=1.1$ to simulate the wealth of the players.

# Valuation Matrix (2D)
To represent the **256 GB** memory in granularity of **64 MB**, we need 4K allocation units.  
To represent the **64 CPUs** in granularity of **$\frac1{64}$ CPU**, we also need 4K allocation units.

We assume the minimum allocation is **t2.nano**, hence, depending on the amount of **t2.nano** players, the rest of
the resources are for auction.  
Thus, the final valuation matrix will be of size: $\frac{256GB-\frac12pGB}{64MB} \times 64\cdot(64-\frac18p)$
$\Rightarrow (4096-8p) \times (4096-8p)$.

Each players could be allocated at most **16 GB** of memory and **4 CPUs** ($\frac1{16}$ of the machine).  
Thus, each valuation matrix will be of size $256 \times 256$.
"""


def generate(sd):
    generate_init_data(sd)
    generate_data(sd)


folder_format = "{ndim}d-{n}p"


metadata_nonconcave = {
    'ndim': 6,
    'n': 256,
    'valuation': {
        'wealth-dist': ['lomax', math.log(5, 4), (0, 128)],
    },
}
nonconcave = dataset.alter_dataset(metadata_nonconcave, folder_format, generator_func=generate,
                                   prefix='vcg-nonconcave')

nonrising = dataset.alter_dataset(nonconcave, folder_format, (('valuation', 'local-maximum-limit'), 3),
                                  prefix='vcg-nonrising')

concave = dataset.alter_dataset(nonconcave, folder_format, (('valuation', 'concave'), True),
                                prefix='vcg-concave')

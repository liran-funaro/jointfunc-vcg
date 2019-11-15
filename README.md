# jointfunc-vcg

Using `cloudsim` to run experiments on the jointfunc algorithm.
See the full paper for more information:

Liran Funaro, Orna Agmon Ben-Yehuda, and Assaf Schuster. 2019. Efficient Multi-Resource, Multi-Unit VCG Auction.

# Install (beta)
Download and install dependencies by cloning the following repositories:

 * vecfunc: https://github.com/liran-funaro/vecfunc
 * vecfunc-vcg: https://github.com/liran-funaro/vecfunc-vcg
 * cloudsim: https://github.com/liran-funaro/cloudsim
 
Follow the `README.md` file in each of these repositories to install them properly.
 
Finally, install the package in developer mode:
```bash
python setup.py develop --user
```

# Usage
The notebooks in the [notbooks](notebooks) folder are used to produce the results seen in the paper. 

# License
[GPL](LICENSE.txt)
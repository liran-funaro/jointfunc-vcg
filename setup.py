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
from setuptools import setup

setup(
    name="jointfunc-vcg",
    version="0.1.0",
    packages=['jointfunc_vcg'],
    description="Using `cloudsim` to run experiments on the jointfunc algorithm.",
    author="Liran Funaro",
    author_email="liran.funaro+jointfunc-vcg@gmail.com",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://bitbucket.org/funaro/jointfunc-vcg",
    keywords='cloud vcg jointfunc maille tuffin',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=['numpy', 'matplotlib', 'scipy', 'seaborn', 'pandas', 'vecfunc', 'vecfunc-vcg', 'cloudsim',
                      'tabulate'],
    dependency_links=['https://bitbucket.org/funaro/vecfunc', 'https://bitbucket.org/funaro/vecfunc-vcg',
                      'https://bitbucket.org/funaro/cloudsim'],
)

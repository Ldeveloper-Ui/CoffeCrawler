"""
Setup script for SpeedForce C++ extension with maximum optimization.
Bro thinks he's A flash.
"""

from setuptools import setup, Extension
from pybind11 import get_include
import pybind11
import os
import sys

# Compiler optimization flags
extra_compile_args = [
    '-std=c++17',
    '-O3',  # Maximum optimization
    '-march=native',  # Use native architecture
    '-mtune=native',
    '-fopenmp',  # OpenMP support
    '-fopenmp-simd',
    '-mavx',  # AVX instructions
    '-mavx2',
    '-mfma',
    '-m64',
    '-fPIC',
    '-DNDEBUG'  # No debug
]

extra_link_args = [
    '-fopenmp',
    '-mavx',
    '-mavx2'
]

# Define the quantum extension module
speedforce_module = Extension(
    'SpeedForce',
    sources=['SpeedForce.cpp'],
    include_dirs=[
        get_include(),
        pybind11.get_include(),
    ],
    language='c++',
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='SpeedForce',
    version='1.0.0',
    description='🚀 Ultra-Fast C++ Quantum Accelerator for CoffeCrawler',
    long_description='''
    SpeedForce - Revolutionary C++ acceleration for CoffeCrawler
    Features:
    • AVX-optimized string processing
    • Multi-threaded parallel execution  
    • SIMD-accelerated pattern matching
    • Quantum caching with LRU eviction
    • Nanosecond-precision timing
    • Memory-optimized data structures
    ''',
    ext_modules=[speedforce_module],
    zip_safe=False,
    python_requires='>=3.7',
)

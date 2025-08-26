from sys import platform
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from numpy import get_include
from os.path import join

# =================
# Cython optimizers
# =================
cython_directives = {
    "language_level": 3,    # python3
    "boundscheck": False,   # skip array bounds check
    "wraparound": False,    # prevent negative indexing
    "cdivision": True,      # use c-style division
    "nonecheck": False      # skip null/none value check
}

# ============
# Source Files
# ============
pyx_files = [
    "bgp.pyx",
    "tgp.pyx",
    "tcp.pyx",
    "rastio.pyx",
    "math_utils.pyx",
]

# =========================================
# OpenMP Compiler Flags (platform-specific)
# =========================================
extra_compile_args = ["/openmp"] if platform == "win32" else ["-O3", "-fopenmp"]
extra_link_args = [] if platform == "win32" else ["-fopenmp"]

# =========================
# Extensions List
# =========================
extensions = [
    Extension(
        fn[:-4],                         # module name
        [join("gosp", "src", fn)],       # source location
        include_dirs=[get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    for fn in pyx_files
]

# ===============================
# Setup call
# ===============================
setup(
    name="gosp",
    version="3.2",
    packages=find_packages(where="gosp"),
    package_dir={"": "gosp"},  # tells setuptools the top-level package is under gosp/
    ext_modules=cythonize(extensions, compiler_directives=cython_directives),
    zip_safe=False,
)

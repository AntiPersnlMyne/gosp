#!/usr/bin/env python3

"""main.py: Main logic file for image processing on manuscript data"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger", "Julie Decker"]
__license__ = "MIT"
__version__ = "3.1.1"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Development", or "Production". 

# ---------------
# Useful commands
# ---------------
"""
// (Development) Compile Cython files 
python setup.py build_ext --inplace

// (Install) Build & Compile Cython files
pip install -e . && rm -r build || del build && rm -r gosp/gosp.egg-info || del gosp/gosp.egg-info
"""


# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from gosp import gosp

import pstats, cProfile

from time import time
from warnings import filterwarnings


# GeoTIFF warning suppression
filterwarnings("ignore", category=UserWarning, message="Dataset has no geotransform, gcps, or rpcs.*")



# --------------------------------------------------------------------------------------------
# Driver Code
# --------------------------------------------------------------------------------------------
def main():
    start = time()
    gosp(
        # Input information
        input_dir="data/input/test",   
        output_dir="data/output",
        input_image_types="tif",
        # BGP and TCP parameters
        full_synthetic=True,                   
        skip_bgp=False,                 
        max_targets=3,                
        opci_threshold=0.01,              
        # Throughput
        window_shape=(1024,1024),                 
        # Debug
        verbose=True,                      
    )
    print(f"\n[main/arch177_rgb_365cor_lum] - Execution finished -\nRuntime = {(time() - start):.2f}")
    


# ================
# Executing Driver
# ================
if __name__ == "__main__":    
    # Memory/Performance profiler
    
    cProfile.runctx("main()", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()


    # # Regular execution
    # main()





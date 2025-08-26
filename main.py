#!/usr/bin/env python3

"""main.py: Main logic file for image processing on manuscript data"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger", "Julie Decker"]
__license__ = "MIT"
__version__ = "4.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Production" # "Development", or "Production". 

# ---------------
# Useful commands
# ---------------
"""
// (Setup, Linux) Build & Compile Cython files
pip install -e . && rm -r build && rm -r gosp/gosp.egg-info 

// (Setup, Windows) Build & Compile Cython files
pip install -e . && del build && del gosp/gosp.egg-info

// (Run) Set # of available threads for multithreading. Depends on your CPU model.
OMP_NUM_THREADS=12 python main.py
"""


# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from gosp import gosp

from time import time
from warnings import filterwarnings


# --------------------------------------------------------------------------------------------
# Driver Code
# --------------------------------------------------------------------------------------------
# GeoTIFF warning suppression
filterwarnings("ignore", category=UserWarning, message="Dataset has no geotransform, gcps, or rpcs.*")

def main():
    start = time()
    gosp(
        # Input information
        input_dir="data/input/arch177_rgb_365cor_lum",   
        output_dir="/media/g-m/ExtremeSan",
        input_image_types="tif",
        # BGP and TCP parameters
        full_synthetic=True,            
        max_targets=50,                
        opci_threshold=0.001,              
        # Throughput
        window_shape=(1024,1024),                 
        # Debug
        verbose=True,                      
    )
    print(f"\n[main/<data_description>] - Execution finished -\nRuntime = {(time() - start):.2f}")
    


# ================
# Executing Driver
# ================
if __name__ == "__main__":    
    main()





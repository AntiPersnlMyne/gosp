#!/usr/bin/env python3

"""gosp_pipeline.py: GOSP (Generalized Orthogonal Subspace Projection)

Implements: The OSP-based, automatic target detection workflow laid out by Chang and Ren.
[Ref] Hsuan Ren, Student Member, IEEE, and Chein-I Chang, Senior Member, IEEE 2000

Does: Automatically finds K (an integer > 1) likely targets in image and classififes all pixels to a target likelihood.
                            
Stages:
    0. Compile Cython (.pyx) code
    1. Band Generation Process (BGP)        - Create synthetic bands from raw imagery
    2. Target Generation Process (TGP)      - Iteratively discover target spectra using OSP
    3. Target Classification Process (TCP)  - Classify image using discovered targets
"""

# --------------------------------------------------------------------------------------------
# Imports Pipeline Modules
# --------------------------------------------------------------------------------------------
import logging
from typing import Tuple, List
from glob import glob
from os.path import join
from os import remove
import rasterio
import numpy as np

from ..build.bgp import band_generation_process
from ..build.tgp import target_generation_process
from ..build.tcp import target_classification_process
from ..build.skip_bgp import write_original_multiband


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.3.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


# --------------------------------------------------------------------------------------------
# Input file discovery (Helper Function)
# --------------------------------------------------------------------------------------------
def _discover_image_files(
    input_dir: str,
    input_image_type: str|Tuple[str, ...] = "tif"
) -> List[str]:
    """
    Discovers and returns a list of image files in a directory matching the given type(s).

    Args:
        input_dir (str): Directory to search for input images.
        input_image_type (str | tuple[str, ...]): File extension(s) to include (e.g. "tif" or ("tif", "png"))

    Returns:
        List[str]: Sorted list of full paths to input image files.
    """
    if isinstance(input_image_type, str):
        input_image_type = (input_image_type,)

    input_files = []
    for file_extension in input_image_type:
        input_files.extend(glob(join(input_dir, f"*.{file_extension}")))

    input_files.sort()
    return input_files


# --------------------------------------------------------------------------------------------
# GOSP Pipeline
# --------------------------------------------------------------------------------------------
def gosp(
    input_dir:str, 
    output_dir:str, 
    input_image_types:str|tuple[str, ...] = "tif",
    window_shape:tuple = (512,512),
    full_synthetic:bool = False,
    skip_bgp:bool = False,
    max_targets:int = 10,
    opci_threshold:float = 0.01,
    verbose:bool = False,
) -> None:
    # IO variables
    input_files = _discover_image_files(input_dir, input_image_types)
    targets_classified_dir = f"{output_dir}"
    
    if not input_files:
        raise FileNotFoundError(
            f"No input images found in {input_dir} with extension(s): {input_image_types}"
        )

    if verbose: logging.basicConfig(level=logging.INFO)
    else:       logging.basicConfig(level=logging.WARNING)


    # Catch any error, delete temporary files to prevent faulty re-executions
    try:
        logging.info("[GOSP] Running Band Generation Process (BGP)...")
        if not skip_bgp:
            band_generation_process(
                input_image_paths=input_files,
                output_dir=output_dir,
                window_shape=window_shape,
                full_synthetic=full_synthetic,
                verbose=verbose
            )
            generated_bands = _discover_image_files(output_dir, "tif")  # collect output bands
        else:
            write_original_multiband(
                input_image_paths=input_files,
                output_dir=output_dir,
                window_shape=window_shape,
                verbose=verbose
            )
            generated_bands = _discover_image_files(output_dir, "tif")



        # Building vrt here prevents race condition in rastio
        with rasterio.open(generated_bands[0]) as src:
            vrt = src.read().astype(np.float32)  # shape (bands, rows, cols)



        logging.info("[GOSP] Running Target Generation Process (TGP)...")
        targets: np.ndarray = target_generation_process(
            vrt,
            window_shape,
            max_targets,
            opci_threshold,
            verbose
        )
        logging.info(f"[GOSP] TGP detected {len(targets)} target(s).")



        logging.info("[GOSP] Running Target Classification Process (TCP)...")
        target_classification_process(
            generated_bands=generated_bands,
            window_shape=window_shape,
            targets=targets,
            output_dir=targets_classified_dir,
            verbose=verbose
        )
        logging.info(f"[GOSP] Complete. Results written to: {targets_classified_dir}")
    
    # Catch and display any errors
    except Exception as e:
        raise Exception(f"[GOSP] -- Error while running GOSP --:\n{e}")    
    
    # Cleanup temporary file always
    finally:
        remove(f"{output_dir}/gen_band_norm.tif")


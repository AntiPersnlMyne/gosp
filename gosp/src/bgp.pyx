#!/usr/bin/env python3
# distutils: language = c
# cython: profile=True

"""bgp.pyx: Band Generation Process, creates new non-linear bondinations of existing bands"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np

from typing import Tuple, List
from logging import info
from tqdm import tqdm

from cython.parallel import prange
from libc.math cimport sqrtf, log1pf

from gosp.build.rastio import MultibandBlockReader, MultibandBlockWriter


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.2.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
np.import_array()

# Typed aliases for readability
ctypedef np.float32_t float32_t


# --------------------------------------------------------------------------------------------
# C Functions
# --------------------------------------------------------------------------------------------
cdef inline Py_ssize_t total_bands(
    Py_ssize_t nbands,
    bint full_synthetic
) noexcept nogil:
    """ Returns total number of bands that will be created"""
    cdef Py_ssize_t total = nbands + (nbands * (nbands - 1)) // 2
    if full_synthetic:
        total += 2 * nbands
    return total


cdef void create_bands_from_block(
    float32_t[:, :, :] src,        # (bands, h, w)
    float32_t[:, :, :] out,        # (out_bands, h, w)
    bint full_synthetic
) noexcept nogil:
    """
    Fills out with synthetic bands:
      - Auto correlation
      - Cross correlation
      - (optional) sqrt and ln 
    """
    cdef:
        Py_ssize_t b, i, j, row, col
        Py_ssize_t bands = src.shape[0]
        Py_ssize_t height = src.shape[1]
        Py_ssize_t width  = src.shape[2]
        Py_ssize_t dst = 0
        float32_t v

    # Original
    for b in range(bands):
        for row in prange(height, nogil=True, schedule="static"):
            for col in range(width):
                out[dst + b, row, col] = src[b, row, col]
    dst += bands

    # Pairwise products
    for i in range(bands - 1):
        for j in range(i + 1, bands):
            for row in prange(height, nogil=True, schedule="static"):
                for col in range(width):
                    out[dst, row, col] = src[i, row, col] * src[j, row, col]
            dst += 1

    if full_synthetic:
        # sqrt
        for b in range(bands):
            for row in prange(height, nogil=True, schedule="static"):
                for col in range(width):
                    v = src[b, row, col]
                    out[dst + b, row, col] = sqrtf(v) if v >= 0.0 else 0.0
        dst += bands

        # log1p
        for b in range(bands):
            for row in prange(height, nogil=True, schedule="static"):
                for col in range(width):
                    v = src[b, row, col]
                    out[dst + b, row, col] = log1pf(-0.99) if v <= -1.0 else log1pf(v)
        dst += bands


cdef void block_minmax(
    float32_t[:, :, :] block,
    float32_t[:] band_mins,
    float32_t[:] band_maxs
) noexcept nogil:
    """ Lightweight pass, gathering of min/max value of each band """ 
    cdef Py_ssize_t b, row, col
    cdef Py_ssize_t bands = block.shape[0]
    cdef Py_ssize_t height = block.shape[1]
    cdef Py_ssize_t width  = block.shape[2]
    cdef float32_t v, local_min, local_max
    for b in prange(bands, nogil=True, schedule="static"):
        local_min = band_mins[b]
        local_max = band_maxs[b]
        for row in range(height):
            for col in range(width):
                v = block[b, row, col]
                if v < local_min:
                    local_min = v
                elif v > local_max:
                    local_max = v
        band_mins[b] = local_min
        band_maxs[b] = local_max


cdef void normalize_block(
    float32_t[:, :, :] block,
    float32_t[:] band_mins,
    float32_t[:] band_maxs
) noexcept nogil:
    """ Uses the found min/max to normalize to float32 [0,1] range """
    cdef Py_ssize_t b, row, col
    cdef Py_ssize_t bands = block.shape[0]
    cdef Py_ssize_t height = block.shape[1]
    cdef Py_ssize_t width  = block.shape[2]
    cdef float32_t denom, val

    for b in prange(bands, nogil=True, schedule="static"):
        denom = band_maxs[b] - band_mins[b]
        if denom == 0.0:
            for row in range(height):
                for col in range(width):
                    block[b, row, col] = 0.0
        else:
            for row in range(height):
                for col in range(width):
                    val = block[b, row, col]
                    block[b, row, col] = (val - band_mins[b]) / denom



# --------------------------------------------------------------------------------------------
# Band Generation Process (2-pass, single output)
# --------------------------------------------------------------------------------------------
def band_generation_process(
    input_image_paths:List[str],
    output_dir:str,
    window_shape:Tuple[int, int],
    full_synthetic:bool,
    verbose:bool=True
) -> None:
    """
    Two pass system:
      - Pass 1: stats-only (global min/max per band)
      - Pass 2: regenerate + normalize, write directly to final file
    """
    cdef bint full_syn = <bint> full_synthetic
    cdef int img_h, img_w

    # Inspect image dims + output bands
    with MultibandBlockReader(input_image_paths) as reader:
        img_h, img_w = reader.get_image_shape()
        test_block = np.ascontiguousarray(reader.read_multiband_block(np.array([0,0,1,1], dtype=np.int32)), dtype=np.float32)
    num_bands_out = total_bands(test_block.shape[0], full_syn)
    # Free temp memory
    del test_block

    # Allocate global stats
    band_mins = np.full(num_bands_out, np.inf, dtype=np.float32)
    band_maxs = np.full(num_bands_out, -np.inf, dtype=np.float32)
    cdef float32_t[:] mins_mv = band_mins
    cdef float32_t[:] maxs_mv = band_maxs

    # Pass 1: stats collection
    if verbose: info("[BGP] Pass 1: scanning for min/max ...")
    with MultibandBlockReader(input_image_paths) as reader:
        prog = tqdm(total=reader.num_windows(window_shape), desc="[BGP] Pass 1 stats", disable=not verbose, colour="BLUE")
        for win in reader.generate_windows(window_shape):
            block = np.ascontiguousarray(reader.read_multiband_block(win), dtype=np.float32)
            out = np.empty((num_bands_out, block.shape[1], block.shape[2]), dtype=np.float32)
            # Create bands
            create_bands_from_block(block, out, full_syn)
            # Grab statistics from block
            block_minmax(out, mins_mv, maxs_mv)
            prog.update(1)
        prog.close()

    # Pass 2: regenerate + normalize, direct write
    if verbose: info("[BGP] Pass 2: regenerating + normalizing ...")
    with MultibandBlockReader(input_image_paths) as reader, \
        MultibandBlockWriter(
            output_dir, (img_h, img_w), "gen_band_norm.tif", 
            window_shape, np.float32, num_bands_out) \
        as writer:
            prog = tqdm(total=reader.num_windows(window_shape), desc="[BGP] Pass 2 write", disable=not verbose, colour="CYAN")
            for win in reader.generate_windows(window_shape):
                block = np.ascontiguousarray(reader.read_multiband_block(win), dtype=np.float32)
                out = np.empty((num_bands_out, block.shape[1], block.shape[2]), dtype=np.float32)
                create_bands_from_block(block, out, full_syn)
                normalize_block(out, mins_mv, maxs_mv)
                writer.write_block(out, win)
                prog.update(1)
            prog.close()
    
    return None
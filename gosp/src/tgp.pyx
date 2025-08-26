#!/usr/bin/env python3
# distutils: language=c


"""tgp.pyx: Target Generation Process. Automatically creates N most significant targets in target detection for pixel classification"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np
from cython.parallel import prange

from logging import info
from tqdm import tqdm

from ..build.math_utils import compute_orthogonal_complement_matrix, compute_opci



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
# C Functions
# --------------------------------------------------------------------------------------------
cdef void _extract_window(
    int row_start,
    int row_end,
    int col_start,
    int col_end,
    np.float32_t[:,:,:] vrt,
    np.float32_t[:,:] out_window
) noexcept nogil:
    """
    Copy a sub-image from vrt into out_window.

    Parameters
    ----------
    row_start, row_end : int
        Pixel indices of the vertical window (half-open: [row_start, row_end))
    col_start, col_end : int
        Pixel indices of the horizontal window (half-open)
    vrt : float[:,:,:]
        The synthetic image with shape (bands, height, width)
    out_window : ndarray[float32, ndim=2]
        Pre-allocated buffer array with shape ((row_end-row_start)*(col_end-col_start), bands)
    """
    cdef:
        Py_ssize_t band
        Py_ssize_t row, col
        Py_ssize_t out_row = 0
        Py_ssize_t rows = row_end - row_start
        Py_ssize_t cols = col_end - col_start
        Py_ssize_t bands = vrt.shape[0]

    for row in range(rows):
        for col in range(cols):
            for band in range(bands):
                out_window[out_row, band] = vrt[band, row_start + row, col_start + col]
            out_row += 1


cdef void _best_target(
    np.float32_t[:,:] pixel_window,     # (num_pixels, bands)
    np.float32_t[:,:] orthog_complement,# (bands, proj_dim) -- here proj_dim may be bands if P_perp is full
    np.float32_t[:]   best_target,      # (bands,)
    float *           best_norm_ptr     # pointer to a single float to store best norm
) noexcept nogil:
    """
    For every pixel in *pixel_window* compute its projection onto the
    orthogonal complement defined by *orthog_complement* and keep the
    pixel with the largest projection norm.

    IMPORTANT: this computes projection norms on-the-fly and does not
    allocate a big `projected` buffer.
    """
    cdef:
        Py_ssize_t pixel, band, j, k
        Py_ssize_t num_pixels = pixel_window.shape[0]
        Py_ssize_t bands = pixel_window.shape[1]
        Py_ssize_t proj_dim = orthog_complement.shape[1]

        np.float32_t tmp
        np.float32_t acc_j
        np.float32_t best_local_norm = -1e36
        int best_idx = -1
        np.float32_t norm

    # For each pixel compute norm = sum_j ( sum_b pixel[b]*orthog[b,j] )^2
    for pixel in range(num_pixels):
        norm = 0.0
        for j in range(proj_dim):
            acc_j = 0.0
            for band in range(bands):
                acc_j += pixel_window[pixel, band] * orthog_complement[band, j]
            norm += acc_j * acc_j

        if norm > best_local_norm:
            best_local_norm = norm
            best_idx = pixel

    # Copy best pixel into best_target and write its norm
    if best_idx >= 0:
        for band in range(bands):
            best_target[band] = pixel_window[best_idx, band]
        best_norm_ptr[0] = best_local_norm
    else:
        # No pixels will write zeros
        for band in range(bands):
            best_target[band] = 0.0
        best_norm_ptr[0] = 0.0



# --------------------------------------------------------------------------------------------
# TGP Function
# --------------------------------------------------------------------------------------------
def target_generation_process(
    np.float32_t[:,:,::1] vrt,
    tuple window_shape,
    int max_targets,
    float opci_threshold,
    bint verbose
) -> np.ndarray[np.float32_t]:
    """
    Iterative TGP that implements the Ren & Chang OSP loop.

    Parameters
    ----------
    vrt : ndarray[float32, ndim=3]
        Synthetic image with shape (bands, height, width)
    window_shape : tuple (h, w)
        Size of scanning window in pixels
    max_targets : int
        Maximum number of targets to find
    opci_threshold : float
        Stop if selected candidate OPCI < opci_threshold
    verbose : bool
        Enable progress messages

    Returns
    -------
    ndarray[float32, ndim=2]
        Targets stacked as (K, bands)
    """
    cdef:
        Py_ssize_t height = vrt.shape[1]
        Py_ssize_t width = vrt.shape[2]
        Py_ssize_t bands = vrt.shape[0]
        int window_height = <int> window_shape[0]
        int window_width  = <int> window_shape[1]

        Py_ssize_t rows_per_window = (height + window_height - 1) // window_height
        Py_ssize_t cols_per_window = (width  + window_width  - 1) // window_width
        Py_ssize_t total_windows   = rows_per_window * cols_per_window
        Py_ssize_t num_pixels_local

        # precompute max pixels per window
        Py_ssize_t max_pixels = window_height * window_width

        np.float32_t[:, ::1] best_pixels_mv
        np.float32_t[::1]   best_norms_mv
        cdef int[:, ::1] windows_mv

    # Preallocate flattened window (max size) -- done once in Python space
    cdef np.ndarray[np.float32_t, ndim=2] flattened_window = np.empty((max_pixels, bands), dtype=np.float32)
    cdef np.float32_t[:, ::1] flat_mv = flattened_window

    # Precreate a buffer for an individual best pixel; we'll store per-window results into arrays
    # We'll allocate per-iteration arrays for best per-window results (since proj-dim changes)
    cdef:
        Py_ssize_t row_offset, col_offset, window_index
        int row_start, row_end, col_start, col_end
        Py_ssize_t i_win
        np.float32_t[:, ::1] P_mv

    # Build windows index array in Python (no nogil) so we can iterate or parallelize easily
    # windows_arr shape (total_windows, 4) with (row_off, col_off, height, width)
    cdef np.ndarray[np.int32_t, ndim=2] windows_arr = np.empty((total_windows, 4), dtype=np.int32)
    cdef int wrow, wcol, widx = 0
    for wrow in range(rows_per_window):
        row_offset = wrow * window_height
        row_end_tmp = row_offset + window_height
        for wcol in range(cols_per_window):
            col_offset = wcol * window_width
            col_end_tmp = col_offset + window_width
            # compute actual sizes (clamp at edges)
            rlen = window_height if row_offset + window_height <= height else height - row_offset
            clen = window_width  if col_offset + window_width  <= width  else width  - col_offset
            windows_arr[widx, 0] = <np.int32_t> row_offset
            windows_arr[widx, 1] = <np.int32_t> col_offset
            windows_arr[widx, 2] = <np.int32_t> rlen
            windows_arr[widx, 3] = <np.int32_t> clen
            widx += 1

    # List to hold discovered target vectors (Python space)
    targets = []

    # Main iterative loop: discover up to max_targets
    if verbose:  outer_prog = tqdm(range(max_targets), desc="[TGP] Iterations", colour="MAGENTA", disable= not verbose)
    else:  outer_prog = range(max_targets)

    for k_target in outer_prog:
        # Compute orthogonal complement (GIL must be held)
        if k_target == 0:
            # Identity projector -> represented as full (bands x bands) identity matrix
            P_perp = np.eye(bands, dtype=np.float32)
        else:
            # compute orthogonal complement projector using Python routine
            # compute_orthogonal_complement_matrix expects list of 1D numpy arrays
            P_perp = compute_orthogonal_complement_matrix(targets).astype(np.float32, copy=False)

        # Prepare memoryviews for P_perp
        P_mv = P_perp

        # Arrays to hold per-window best pixel and norm
        best_pixels = np.empty((total_windows, bands), dtype=np.float32)
        best_norms  = np.empty((total_windows,), dtype=np.float32)

        best_pixels_mv = best_pixels
        best_norms_mv  = best_norms

        # NOTE: this is a heavy, pure C-level kernel, with GIL
        for _ in prange(total_windows, nogil=True, schedule='dynamic'):
            # placeholder required by Cython for flow -- actual loop below done in a pythonic wrapper
            pass  

        # Memoryview of windows
        windows_mv = windows_arr  # shape (total_windows, 4)

        # Iterate through windows and gen targets
        num_pixels_local
        for i_win in prange(total_windows, nogil=True, schedule='dynamic'):
            # read the window parameters (pure C-level reads)
            row_offset = windows_mv[i_win, 0]
            col_offset = windows_mv[i_win, 1]
            row_start = row_offset
            col_start = col_offset
            row_end = row_offset + windows_mv[i_win, 2]
            col_end = col_offset + windows_mv[i_win, 3]

            # Extract window into flat_mv (writes first num_pixels entries)
            _extract_window(row_start, row_end, col_start, col_end, vrt, flat_mv)

            # Number of pixels in this window
            num_pixels_local = (row_end - row_start) * (col_end - col_start)

            # Create a sliced view for the active pixels (this is allowed when passing inline)
            # Call _best_target on the slice; pass address of best_norms_mv[i_win] as pointer
            _best_target(flat_mv[:num_pixels_local, :], P_mv, best_pixels_mv[i_win], &best_norms_mv[i_win])

        # After nogil section: select best window candidate (GIL regained)
        # Find index of max norm
        best_idx = int(np.argmax(best_norms))
        best_candidate = best_pixels[best_idx]   # numpy 1D array (bands,)

        # Compute OPCI for the candidate; compute_opci expects P_perp and candidate vector
        opci_val = compute_opci(P_perp, best_candidate)

        info(f"[TGP] Iter {k_target+1}: OPCI={opci_val:.6f}")

        # Stopping condition
        if opci_val < opci_threshold:
            info(f"[TGP] Stopping: OPCI {opci_val:.6f} < threshold {opci_threshold}")
            break

        # Accept target: append to list
        targets.append(best_candidate.copy())

    # End iterations

    # Prepare return array: stack targets into (K, bands)
    if len(targets) == 0:
        # Return empty array with shape (0, bands)
        return np.empty((0, bands), dtype=np.float32)
    else:
        targets_arr = np.stack(targets, axis=0).astype(np.float32, copy=False)
        return targets_arr


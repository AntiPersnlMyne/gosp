#!/usr/bin/env python3
# distutils: language=c
# cython: profile=True

"""math_utils.pyx: Linear algebra, matrix, and calculus helper functions"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
cimport numpy as np

from libc.math cimport sqrt as csqrt
from cython.parallel import prange


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.1.3"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Production" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Constants (to prevent DIV0)
# --------------------------------------------------------------------------------------------
OPCI_EPS = 1e-12   # denom floor
OPCI_TOL = 1e-9    # clamp tolerance


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
ctypedef np.float32_t float_t
ctypedef Py_ssize_t psize_t


# --------------------------------------------------------------------------------------------
# C Helper Functions
# --------------------------------------------------------------------------------------------
cdef void _matvec(
    const float_t[:, :] pmat_mv,
    const float_t[:]    x_mv,
          float_t[:]    y_mv,
) noexcept nogil:
    """
    Compute y = M @ x where M is (n,n), x is (n,), y is (n,).
    All arrays are float32 memoryviews.
    """
    cdef:
        psize_t n = pmat_mv.shape[0]
        psize_t i, j
        float_t sum
    for i in range(n):
        sum = 0.0
        for j in range(n):
            sum += pmat_mv[i, j] * x_mv[j]
        y_mv[i] = sum


cdef void _project_block(
    const float_t[:, :, :] block_mv, # (bands, h, w)
    const float_t[:, :]    P_mv,     # (bands, bands)
          float_t[:, :, :] out_mv    # (bands, h, w)
) noexcept nogil:
    """
    Project a block into a subspace defined by P (bands x bands).


    Args:
    block_mv: input block (bands, h, w)
    P_mv: projection matrix (bands, bands)
    out_mv: preallocated output (bands, h, w)
    """
    cdef:
        psize_t bands = block_mv.shape[0]
        psize_t height = block_mv.shape[1]
        psize_t width = block_mv.shape[2]
        psize_t row, col, b, k

    # Parallelize over rows 
    for row in prange(height, nogil=True, schedule='static'):
        for col in range(width):
            # Compute P @ block[:, row, col] => out[:, row, col]
            for b in range(bands):
                out_mv[b,row,col] = 0.0 # sanity, set to 0
                for k in range(bands):
                    out_mv[b, row, col] += P_mv[b, k] * block_mv[k, row, col]
                

def compute_orthogonal_complement_matrix(
    target_vectors:list[np.ndarray]
) -> np.ndarray:
    """
    Construct an orthogonal projection matrix onto the complement of the
    subspace spanned by given target vectors.


    P_perp = I - U U^+ , where U stacks target_vectors columnwise.


    Args:
    target_vectors (list[np.ndarray]): list of 1D arrays, each (bands,).


    Returns:
        np.ndarray: (bands, bands) float32 symmetric projector.
    """
    if len(target_vectors) == 0:
        raise ValueError("Must provide at least one target vector")

    # Stack in double precision for stable pinv
    U = np.stack(target_vectors, axis=1).astype(np.float64, copy=False)
    P = U @ np.linalg.pinv(U)
    B = U.shape[0]
    I = np.eye(B, dtype=np.float64)
    P_perp = I - P
    P_perp = 0.5 * (P_perp + P_perp.T)
    # Force float32 and return
    return P_perp.astype(np.float32, copy=False)


def project_block_onto_subspace(
    block: np.ndarray,
    proj_matrix: np.ndarray|None
) -> np.ndarray:
    """
    Projects every pixel in a block into the orthogonal subspace defined by the projection matrix.

    Args:
        block (np.ndarray):
            Input block of shape (bands, height, width)
        projection_matrix (np.ndarray): 
            Projection matrix of shape (bands, bands)

    Returns:
        np.ndarray: Projected block of same shape as block (bands, height, width)
    """
    # 1 target ("None") = Identity matrix ("block") 
    if proj_matrix is None: return block

    if block.ndim != 3: raise ValueError("block must be 3D (bands, h, w)")

    # Enforce float32 contiguous
    block_f = np.ascontiguousarray(block, dtype=np.float32)
    P = np.ascontiguousarray(proj_matrix, dtype=np.float32)
    # Symmetrize once (idempotent)
    P = 0.5 * (P + P.T)

    cdef:
        float_t[:, :, :] block_mv = block_f
        float_t[:, :] P_mv = P
        np.ndarray[np.float32_t, ndim=3] out_f = np.empty_like(block_f, dtype=np.float32)
        float_t[:, :, :] out_mv = out_f

    with nogil:
        _project_block(block_mv, P_mv, out_mv)

    return out_f


def compute_opci(
    p_matrix: np.ndarray,
    spectrum: np.ndarray
) -> float:
    """
    Compute Orthogonal Projection Contrast Index (OPCI).
    OPCI = sqrt( (x^T P x) / (x^T x) ).

    Args:
        p_matrix (np.ndarray): (bands, bands) orthogonal complement matrix.
        spectrum (np.ndarray): 1D spectral vector (bands,).


    Returns:
        float: OPCI value in [0,1].
    """
    # Convert to a 1D contiguous vector 
    x_vec = np.asarray(spectrum, dtype=np.float32).reshape(-1)

    # Replace NaNs/Infs with zero
    if not np.isfinite(x_vec).all():
        x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute denominator: ||x||^2
    denom_energy = np.dot(x_vec, x_vec).item()
    # Very small or zero energy vector — nothing to project.
    if denom_energy <= OPCI_EPS:
        return 0.0

    P = np.ascontiguousarray(p_matrix, dtype=np.float32)
    P = 0.5 * (P + P.T)

    y_vec = np.empty_like(x_vec, dtype=np.float32)

    cdef:
        float_t[:, :] P_mv = P
        float_t[:] x_mv = x_vec
        float_t[:] y_mv = y_vec

    with nogil:
        _matvec(P_mv, x_mv, y_mv)

    numerator_energy = np.dot(x_vec, y_vec).item()
    opci = numerator_energy / denom_energy


    if not np.isfinite(opci):
        return 0.0
    if opci < -OPCI_TOL:
        return 0.0
    if opci > 1.0 + OPCI_TOL:
        return 1.0

    cdef float_t opci_clamped = <float> min(max(opci, 0.0), 1.0)
    return <float_t> csqrt(opci_clamped)


#!/usr/bin/env python3
# distutils: language=c

"""rastio.pyx: Handles I/O of raster data"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np
from os import makedirs
from os.path import join

from osgeo import gdal
import rasterio
from rasterio.windows import Window


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "4.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Production" # "Prototype", "Development", "Production"


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
np.import_array()

ctypedef np.float32_t float_t
ctypedef np.uint16_t uint16_t


# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------
def _build_vrt(vrt_path:str, filepaths:list[str], separate=True, allow_projection_difference=True) -> object:
    """
    Create a virtual raster (VRT) from a set of input raster files.
    
    args:
        out_dir (str): 
            Path to the output VRT file (filename not included).
        input_pattern (str): 
            Glob pattern for input raster files.
        separate (bool): 
            If True, stack bands separately.
        allow_projection_difference (bool): 
            If True, allow rasters with different projections.
    
    Returns:
        object: VRT dataset object.
    """
    assert filepaths and len(filepaths) > 0, \
        f"[rastio] No files in filepaths"

    # Build VRT options
    vrt_options = gdal.BuildVRTOptions(
        separate=separate,
        allowProjectionDifference=allow_projection_difference
    )

    # Create the VRT
    vrt = gdal.BuildVRT(
        destName=vrt_path, 
        srcDSOrSrcDSTab=filepaths, 
        options=vrt_options
    )
    
    if vrt is None:
        raise RuntimeError("[rastio] Failed to build VRT")

    return vrt


# --------------------------------------------------------------------------------------------
# Reader (Input)
# --------------------------------------------------------------------------------------------
cdef class MultibandBlockReader:
    """
    A class for reading multi-band raster datasets in blocks (windows).
    Supports both single-band files and multiband files.

    Attributes:
        filepaths (list): 
            A list of paths to the raster file.
        total_bands (int):
            Aggregate number of bands from all files
        win_shape (tuple):
            Window (win_height, win_width)
    """
    cdef:
        object dataset
        str vrt_path
        int total_bands
        tuple image_shape
        list filepaths
        int rasterX, rasterY
    
    def __cinit__(self, list filepaths):
        """
        Initializes the reader.

        Args:
            filepaths (List[str]): A list of path(s) to the raster files.
        """
        self.filepaths = filepaths
        self.vrt_path = "/vsimem/vrt_dataset.vrt"
        self.dataset = _build_vrt(
            vrt_path=self.vrt_path,
            filepaths=filepaths)
        self.total_bands = self.dataset.RasterCount
        self.rasterX = self.dataset.RasterXSize
        self.rasterY = self.dataset.RasterYSize
        self.image_shape = (self.rasterY, self.rasterX)
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
            
    def __del__(self):
        self.close()

    def get_image_shape(self):
        return self.image_shape

    def get_total_bands(self):
        return self.total_bands

    def close(self):
        """Safely close dataset and free VRT path if present."""
        try:
            if self.dataset is not None:
                self.dataset = None
            if self.vrt_path is not None:
                gdal.Unlink(self.vrt_path)
        except Exception:
            pass


    def read_multiband_block(self, np.ndarray[int, ndim=1] window):
        """
        Reads a block of data and returns (bands, rows, cols)
        
        Parameters
        ----------
            window (WindowType): Region of raster to pull data: ( row_off, col_off, win_height, win_width )

        Returns
        ----------
            np.ndarray: A multiband Numpy array representing the block of data 
                with shape (bands, height, width), where (height, width) defined by window.
        """
        cdef:
            int row_off     = <int>window[0]
            int col_off     = <int>window[1]
            int win_h       = <int>window[2]
            int win_w       = <int>window[3]

            np.ndarray[float_t, ndim=3] block 

        # Prevent out of bounds
        row_off = max(0, min(row_off, self.rasterY - 1))
        col_off = max(0, min(col_off, self.rasterX - 1))
        win_h   = min(win_h, self.rasterY - row_off)
        win_w   = min(win_w, self.rasterX - col_off)
                    
        # Read all bands in one shot (as bytes)
        rast_data = self.dataset.ReadRaster(
            col_off, row_off, win_w, win_h,
            buf_xsize=win_w, buf_ysize=win_h,
            buf_type=gdal.GDT_Float32
        )

        # Convert bytes (buffer) to NumPy array and reshape
        block = np.frombuffer(rast_data, dtype=np.float32
            ).reshape((self.total_bands, win_h, win_w)
            ).copy() # shallow copy is Python readable

        return block


    def generate_windows(self, tuple window_shape):
        """
        Yield windows as int32 arrays: [row_off, col_off, win_h, win_w].
        """
        cdef int win_h = <int> window_shape[0]
        cdef int win_w = <int> window_shape[1]
        if win_h <= 0 or win_w <= 0:
            raise ValueError("[rastio] window_shape must be positive")


        cdef int n_rows = (self.rasterY + win_h - 1) // win_h
        cdef int n_cols = (self.rasterX + win_w - 1) // win_w
        cdef int row, col, row_off, col_off, h, w


        for row in range(n_rows):
            row_off = row * win_h
            h = win_h if row_off + win_h <= self.rasterY else self.rasterY - row_off
            for col in range(n_cols):
                col_off = col * win_w
                w = win_w if col_off + win_w <= self.rasterX else self.rasterX - col_off
                yield np.array([row_off, col_off, h, w], dtype=np.int32)


    cpdef int num_windows(self, tuple window_shape):
        """Return total number of windows for a given (win_h, win_w)."""
        cdef int win_h = <int> window_shape[0]
        cdef int win_w = <int> window_shape[1]
        cdef int n_rows = (self.rasterY + win_h - 1) // win_h
        cdef int n_cols = (self.rasterX + win_w - 1) // win_w
        return n_rows * n_cols

# --------------------------------------------------------------------------------------------
# Writer (Output)
# --------------------------------------------------------------------------------------------
cdef class MultibandBlockWriter:
    """
    A class that handles writing blocks of data to an output raster file.

    Attributes
    ----------
        out_dir (str): 
            The path to the output raster directory (filename not included).
        out_image_shape (tuple): 
            The dimensions (rows, cols) of the output image.
        out_image_name (str):
            filename.ext of output file. E.g., `raster.tif`.
        win_shape (tuple):
            Window dimensions (height, width).
        out_dtype (np.type, optional): 
            The data type of the output raster. Defaults to np.float32.
        num_bands (int):
            Number of output bands.
        compress_zstd (bint):
            Compresses output file with ZSTD compression. Smaller file = slower IO speed.
    """
    cdef:
        str out_dir
        tuple out_image_shape
        str out_image_name
        tuple win_shape
        object out_datatype
        int num_bands
        object dataset
        dict profile
    
    def __cinit__(
        self, 
        str output_dir, 
        tuple output_image_shape, 
        str output_image_name, 
        tuple window_shape, 
        object output_datatype,
        int num_bands,
    ):
        self.out_dir         = output_dir
        self.out_image_shape = output_image_shape
        self.out_image_name  = output_image_name
        self.win_shape       = window_shape
        self.out_datatype    = output_datatype
        self.num_bands       = num_bands
        self.dataset         = None
        self.profile         = {}


    def __enter__(self):      
        # Build rasterio profile using the attributes set above
        self.profile = {
            "driver": "GTiff",
            "height": int(self.out_image_shape[0]),
            "width": int(self.out_image_shape[1]),
            "count": int(self.num_bands),
            # rasterio expects np.dtype or string
            "dtype": self.out_datatype if isinstance(self.out_datatype, np.dtype) else np.dtype(self.out_datatype).name,
            "tiled": True,
            "blockxsize": int(self.win_shape[1]),
            "blockysize": int(self.win_shape[0]),
            "interleave": "band",
            "BIGTIFF": "YES",
            "compress": None,
        }

        # Check for valid output path for intermediate dataset file,
        # otherwise create it
        makedirs(self.out_dir, exist_ok=True) 
        out_path = join(self.out_dir, self.out_image_name)
        self.dataset = rasterio.open(out_path, "w", **self.profile)
        return self 

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 

    def close(self):
        try:
            if self.dataset: self.dataset.close()
        except: 
            if self.dataset is not None: self.dataset = None
            pass

    def write_block(
        self, 
        np.ndarray[float_t, ndim=3] block,
        np.ndarray[int, ndim=1] window, 

    ):
        """
        Write block of data to dataset

        Parameters
        ----------
            window (tuple[tuple, tuple]): 
                Section of output dataset to write block to. shape=( (row_off, col_off), (win_height, win_width) )
            block (np.ndarray): 
                Block of data to be written. Size: (bands, win_height, win_width).
        """
        cdef:
            int row_off     = <int>window[0]
            int col_off     = <int>window[1]
            int win_h       = <int>window[2]
            int win_w       = <int>window[3]
        
        # ==============
        # Shape Checking
        # ==============
        if block.dtype != np.float32 or not block.flags['C_CONTIGUOUS']:
            # Ensure correct dtype and contiguous
            block = np.ascontiguousarray(block, dtype=np.float32)

        # Create memoryview
        cdef float_t[:, :, :] block_mv = block

        # Check dims and dataset
        if block_mv.shape[0] != self.num_bands or block_mv.shape[1] != win_h or block_mv.shape[2] != win_w:
            raise ValueError(f"[rastio] Shape mismatch: got {(block_mv.shape[0], block_mv.shape[1], block_mv.shape[2])}, expected ({self.num_bands}, {win_h}, {win_w})")

        if not self.dataset:
            raise RuntimeError("[rastio] Attempted to write but dataset is not initialized")
        
        # ==============================
        # Write & Return Multiband Block
        # ==============================
        win = Window(col_off, row_off, win_w, win_h) 
        self.dataset.write(block, window=win, indexes=range(1, self.num_bands + 1))



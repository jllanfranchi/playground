"""
Arbitrary-dimensional histogramming on the GPU.
"""


from __future__ import division

import os

import numpy as np
from pycuda import autoinit
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceData

from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.utils.profiler import profile
from pisa.utils.log import logging, set_verbosity


__all__ = ['BinningStruct', 'GPUHist',
           'CUDA_HIST_MOD',
           'EVEN_LIN_SPACING', 'EVEN_LOG_SPACING', 'UNEVEN_SPACING']


EVEN_LIN_SPACING = 0
EVEN_LOG_SPACING = 1
UNEVEN_SPACING = 2


class BinningStruct(object):
    def __init__(self, binning, ftype):
        binning = MultiDimBinning(binning)
        num_dims = np.int32(binning.num_dims)
        bins_per_dim = np.array(shape=num_dims, dtype=np.int32)
        # For dim_spacing_type: 0 => lin, 1 => log, 2 => arbitrary
        dim_spacing_type = np.array(shape=num_dims, dtype=np.uint8)
        further_dims_bincounts = np.array(shape=num_dims, dtype=np.int32)

        bmin = np.array(shape=num_dims, dtype=ftype)
        bmax = np.array(shape=num_dims, dtype=ftype)
        bwidth = np.array(shape=num_dims, dtype=ftype)

        units = []

        # Sequence of arrays
        bin_edges = []

        for dim_num, dim in enumerate(binning.dims):
            bins_per_dim[dim_num] = len(dim)
            units.append(dim.bin_edges.units)

            if dim.is_lin:
                dim_spacing_type[dim_num] = EVEN_LIN_SPACING
                bwidth[dim_num] = (bmax[dim_num] - bmin[dim_num])/len(dim)
                assert bwidth[dim_num] > 0
                bin_edges.append(np.array([np.nan], dtype=ftype))
                bmin[dim_num] = dim.bin_edges[0].magnitude
                bmax[dim_num] = dim.bin_edges[-1].magnitude
            elif dim.is_log:
                dim_spacing_type[dim_num] = EVEN_LOG_SPACING
                # Record critical bin values in log space
                bmin[dim_num] = np.log(dim.bin_edges[0].magnitude)
                bmax[dim_num] = np.log(dim.bin_edges[-1].magnitude)
                bwidth[dim_num] = (bmax[dim_num] - bmin[dim_num])/len(dim)
                assert bwidth[dim_num] > 0
                bin_edges.append(np.array([np.nan], dtype=ftype))
            else:
                dim_spacing_type[dim_num] = UNEVEN_SPACING
                bwidth[dim_num] = np.nan
                bin_edges.append(dim.bin_edges.magnitude.astype(ftype))
                bmin[dim_num] = dim.bin_edges[0].magnitude
                bmax[dim_num] = dim.bin_edges[-1].magnitude

        # Total bincounts in subsequent dimensions for indexing
        cumulative_bincount = 1
        for dim_idx in range(num_dims-1, -1, -1):
            further_dims_bincount[dim_idx] = cumulative_bincount
            cumulative_bincount *= bins_per_dim[dim_idx]

        # Record members of struct here (to be passed to device)
        self.num_dims = np.int32(num_dims)
        self.bins_per_dim = bins_per_dim
        self.dim_spacing_type = dim_spacing_type
        self.further_dims_bincount = further_dims_bincount
        self.bmin = bmin
        self.bmax = bmax
        self.bwidth = bwidth
        self.bin_edges = bin_edges

        # Record things useful to keep around on Python side as "private" attrs
        self._units = units
        self._binning = binning

    @property
    def sizeof(self):

    def struct_to_device(self, device_loc):



    #for (int dim_idx = 0; dim_idx < NUM_DIMS; dim_idx++) {
    #    if (dim_spacing_type[dim_idx] == 1) {
    #        bmin[dim_idx] = binning[dim_idx][0];
    #        bmax[dim_idx] = binning[dim_idx][bins_per_dim[dim_idx]];
    #        bwidth[dim_idx] = (bmax[dim_idx] - bmin[dim_idx])/bins_per_dim[dim_idx];
    #    }
    #    else if (dim_spacing_type[dim_idx] == 2) {
    #        bmin[dim_idx] = log(binning[dim_idx][0]);
    #        bmax[dim_idx] = log(binning[dim_idx][bins_per_dim[dim_idx]]);
    #        bwidth[dim_idx] = log(bin_deltas[dim_idx]);
    #        bwidth[dim_idx] = (log(bmax[dim_idx]) - log(bmin[dim_idx])) / bins_per_dim[dim_idx];
    #    }
    #    else {
    #        bmin[dim_idx] = 0.0;
    #        bmax[dim_idx] = 0.0;
    #        bwidth[dim_idx] = 0.0;
    #    }
    #}


class GPUHist(object):
    def __init__(self, binning, data_units=None):
        self._get_gpu_info()
        self._setup_binning(binning=binning, data_units=data_units)

    def _setup_binning(self, binning, data_units=None):
        self.binning = MultiDimBinning(binning)
        # TODO: do something meaningful with units
        btruct = 
        for dim in self.binning.dims:
        self.

    def _get_gpu_info(self):
        gpu = autoinit.device
        self.gpu = gpu
        self.device_data = DeviceData()

    def _get_cuda_run_params(self, compiled_kernel, num_events):
        """Get "optimal" (or at least dynamic) block & thread parameters for
        running the compiled kernel.

        Returns
        -------
        threads_per_block, num_blocks, events_per_thread : int

        See http://pythonhosted.org/CudaPyInt/_modules/cudapyint/Solver.html

        """
        max_threads = min(
            self.device_data.max_registers_per_block/compiled_kernel.num_regs,
            self.device_data.max_threads
        )
        max_warps = max_threads / self.device_data.warp_size
        threads_per_block = int(np.ceil(max_warps * self.device_data.warp_size))

        # Use only a single multiprocessor (cpu core), so limits the number of
        # thread blocks (?)
        max_thread_blocks = self.device_data.thread_blocks_per_mp 

        num_blocks, r = divmod(num_events, threads_per_block)
        if r != 0:
            num_blocks += 1

        num_blocks = min(num_blocks, max_thread_blocks)

        events_per_thread = int(np.ceil(num_events / (threads_per_block *
                                                      num_blocks)))

        return threads_per_block, num_blocks, events_per_thread


CUDA_HIST_MOD = SourceModule(
"""//CUDA//

/*
 * setup things that will not change across multiple invocations of histogram
 * function
 */
__global__ setup(const int n_dims, const int n_flat_bins, const int num_events)
{
    // Following must be supplied by the caller
    int bins_per_dim[NUM_DIMS];
    int dim_spacing_type[NUM_DIMS];

    // Initialize further_dims_bincounts for indexing into the flattened
    // histogram. One can imagine indexing into hist via e.g. 3 indices if it's
    // 3-dimensional:
    //     hist[i][j][k]
    // with i in [0, I-1], j in [0, J-1], and k in [0, K-1]. This indexes into
    // the linear chunk of memory that stores the histogram in row-major order,
    // i.e., this translates to linear (flattened) memory address
    //     hist[i*J*K + j*K + k]
    // and analogously for lower/higher dimensions.

    int further_dims_bincounts[NUM_DIMS];
    int cumulative_bincount = 1;
    if (NUM_DIMS > 1) {
        for (int dim_idx = NUM_DIMS-1; dim_idx >= 0; dim_idx--) {
            further_dims_bincount[dim_idx] = cumulative_bincount;
            cumulative_bincount *= bins_per_dim[dim_idx];
        }
    }

    // Initialize binning constants needed for fast histogramming of lin/log
    // dims

    fType bmin[NUM_DIMS];
    fType bmax[NUM_DIMS];
    fType bwidth[NUM_DIMS];
    for (int dim_idx = 0; dim_idx < NUM_DIMS; dim_idx++) {
        if (dim_spacing_type[dim_idx] == %(EVEN_LIN_SPACING)d) {
            bmin[dim_idx] = binning[dim_idx][0];
            bmax[dim_idx] = binning[dim_idx][bins_per_dim[dim_idx]];
            bwidth[dim_idx] = (bmax[dim_idx] - bmin[dim_idx])
                              / bins_per_dim[dim_idx];
        }
        else if (dim_spacing_type[dim_idx] == %(EVEN_LOG_SPACING)d) {
            bmin[dim_idx] = log(binning[dim_idx][0]);
            bmax[dim_idx] = log(binning[dim_idx][bins_per_dim[dim_idx]]);
            bwidth[dim_idx] = log(bin_deltas[dim_idx]);
            bwidth[dim_idx] = (log(bmax[dim_idx]) - log(bmin[dim_idx]))
                              / bins_per_dim[dim_idx];
        }
        else { // UNEVEN_SPACING
            bmin[dim_idx] = 0.0;
            bmax[dim_idx] = 0.0;
            bwidth[dim_idx] = 0.0;
        }
    }
}


/*
 * Histogram the D-dimensional data store
 */
__global__ histogramdd(int n_samples, fType *sample, fType *weights)
{
    // TODO:
    // Is this faster    : fType data[NUM_EVENTS][NUM_DIMS+1];
    // Or is this faster : fType data[NUM_DIMS+1][NUM_EVENTS];

    // Note weights are last array stored to data, hence the length NUM_DIMS+1
    fType data[NUM_DIMS+1][NUM_EVENTS];


    // Perform the histogramming operation
    
    fType val;
    int flat_idx;
    int thisdim_bin_idx;
    for (evt_idx = EVT_START_IDX; evt_idx < EVT_STOP_IDX; evt_idx++) {
        flat_idx = 0;
        dim_indices = int[NUM_DIMS];

        // TODO: go through dimensions in order of least-expensive-to-compute
        // (even linear bin spacing) to most-expensive-to-compute (uneven bin
        // spacing)

        for (dim_idx = 0; dim_idx < NUM_DIMS; dim_idx++) {
            if (dim_spacing_type[dim_idx] == %(UNEVEN_SPACING)d) {
                // do binary search (see Philipp's code for this)
            }
            else {
                if (dim_spacing_type[dim_idx] == %(EVEN_LIN_SPACING)d)
                    val = data[dim_idx][evt_idx];
                else
                    val = log(data[dim_idx][evt_idx]);
    
                thisdim_bin_idx = <int> (val - bmin[dim_idx]) / bwidth[dim_idx];
    
                // Move on to the next event if this event doesn't fall within
                // any of this dimension's bins
                if (thisdim_bin_idx < 0
                    || thisdim_bin_idx > bins_per_dim[dim_idx]
                    || (thisdim_bin_idx == bins_per_dim[dim_idx]
                        && val != bmax))
                    break;
            }
    
            // If we get here, then the event *is* in a bin in this dimension.
    
            if (dim_idx == NUM_DIMS-1)
                hist[flat_idx + thisdim_bin_idx] += data[NUM_DIMS][evt_idx];
            else
                flat_idx += thisdim_bin_idx * further_dims_bincount[di];
        }
    }
""" % dict(EVEN_LIN_SPACING=EVEN_LIN_SPACING,
           EVEN_LOG_SPACING=EVEN_LOG_SPACING,
           UNEVEN_SPACING=UNEVEN_SPACING)
)

import math

import dask.array as da
import zarr
import cupy as cp
import numpy as np

import time as times
from cellstar_preprocessor.flows.common import (
    compute_downsamplings_to_be_stored,
    compute_number_of_downsampling_steps,
    open_zarr_structure_from_path,
)
from cellstar_preprocessor.flows.constants import (
    DOWNSAMPLING_KERNEL,
    MIN_GRID_SIZE,
    QUANTIZATION_DATA_DICT_ATTR_NAME,
    VOLUME_DATA_GROUPNAME,
)
from cellstar_preprocessor.flows.volume.helper_methods import (
    generate_kernel_3d_arr,
    gaussian_kernel_3d,
    store_volume_data_in_zarr_stucture,
)
from cellstar_preprocessor.model.volume import InternalVolume
from cellstar_preprocessor.tools.quantize_data.quantize_data import (
    decode_quantized_data,
)

from cupyx.scipy.ndimage import convolve as gpu_convolve

def volume_downsampling_gpu(internal_volume: InternalVolume):
    """
    Do downsamplings, store them in intermediate zarr structure
    Note: takes original data from 0th resolution, time_frame and channel
    """
    
    zarr_structure = open_zarr_structure_from_path(
        internal_volume.intermediate_zarr_structure_path
    )

    original_res_gr: zarr.Group = zarr_structure[VOLUME_DATA_GROUPNAME]["1"]
    for time, timegr in original_res_gr.groups():
        timegr: zarr.Group
        for channel_id, channel_arr in timegr.arrays():
            # NOTE: skipping convolve if one of dimensions is 1
            if 1 in channel_arr.shape:
                print(
                    f"Downsampling skipped for volume channel {channel_id}, timeframe {time}"
                )
                continue

            original_data_arr = zarr_structure[VOLUME_DATA_GROUPNAME]["1"][str(time)][
                str(channel_id)
            ]
            if QUANTIZATION_DATA_DICT_ATTR_NAME in original_data_arr.attrs:
                data_dict = original_data_arr.attrs[QUANTIZATION_DATA_DICT_ATTR_NAME]
                data_dict["data"] = cp.asarray(original_data_arr[:])
                cp_array = decode_quantized_data_gpu(data_dict)
            else:
                cp_array = cp.asarray(original_data_arr[:])

            current_level_data = cp_array
            # 1. compute number of downsampling steps based on internal_volume.downsampling
            # 2. compute list of ratios of downsamplings to be stored based on internal_volume.downsampling
            # 3. if ratio is in list, store it

            # downsampling_steps = 8
            downsampling_steps = compute_number_of_downsampling_steps(
                int_vol_or_seg=internal_volume,
                min_grid_size=MIN_GRID_SIZE,
                input_grid_size=math.prod(cp_array.shape),
                factor=2**3,
                force_dtype=cp_array.dtype,
            )

            ratios_to_be_stored = compute_downsamplings_to_be_stored(
                int_vol_or_seg=internal_volume,
                number_of_downsampling_steps=downsampling_steps,
                input_grid_size=math.prod(cp_array.shape),
                factor=2**3,
                dtype=cp_array.dtype,
            )
            
            gpu_kernel = gaussian_kernel_3d(5, 1.0)
            for i in range(downsampling_steps):
                current_ratio = 2 ** (i + 1)
                
                d_volume = current_level_data
                    
                d_convolved = gpu_convolve(d_volume, gpu_kernel, mode='mirror')
                    
                d_downsampled = d_convolved[::2, ::2, ::2]
                

                if current_ratio in ratios_to_be_stored:
                    store_volume_data_in_zarr_stucture(
                        data=da.from_array(cp.asnumpy(d_downsampled)),
                        volume_data_group=zarr_structure[VOLUME_DATA_GROUPNAME],
                        params_for_storing=internal_volume.params_for_storing,
                        force_dtype=internal_volume.volume_force_dtype,
                        resolution=current_ratio,
                        time_frame=time,
                        channel=channel_id,
                    )

                current_level_data = d_downsampled
                
            del d_volume, d_convolved, d_downsampled
            cp.get_default_memory_pool().free_all_blocks()
            
            print("Volume downsampled")

    # # NOTE: remove original level resolution data
    # if internal_volume.downsampling_parameters.remove_original_resolution:
    #     del zarr_structure[VOLUME_DATA_GROUPNAME]["1"]
    #     print("Original resolution data removed")

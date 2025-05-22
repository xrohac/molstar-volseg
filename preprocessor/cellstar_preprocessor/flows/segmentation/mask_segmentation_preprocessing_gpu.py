from pathlib import Path
import cupy as cp

import mrcfile
import numpy as np
import zarr
from typing import Dict, List, Any
from cellstar_preprocessor.flows.common import (
    open_zarr_structure_from_path,
    set_segmentation_custom_data,
)
from cellstar_preprocessor.flows.constants import LATTICE_SEGMENTATION_DATA_GROUPNAME
from cellstar_preprocessor.flows.segmentation.helper_methods import (
    store_segmentation_data_in_zarr_structure,
)
from cellstar_preprocessor.model.input import SegmentationPrimaryDescriptor
from cellstar_preprocessor.model.segmentation import InternalSegmentation



def gpu_normalize_axis_order(arr: cp.ndarray, current_order: tuple) -> cp.ndarray:
    """
    Normalize axis order using CuPy for GPU-accelerated processing
    """
    if current_order != (0, 1, 2):
        print(f"Reordering axes from {current_order}...")
        ao = {v: i for i, v in enumerate(current_order)}
        return cp.transpose(arr, axes=(ao[2], ao[1], ao[0]))
    else:
        arr = cp.transpose(arr)
    return arr

def get_optimal_chunk_size(Y, X, element_size=4, reserve=0.7):
    free_mem = cp.cuda.Device(0).mem_info[0]
    usable_mem = int(free_mem * reserve)
    print(f"Free memory: {free_mem}, Usable memory: {usable_mem}")
    max_elements = usable_mem // element_size
    chunk_size = max_elements // (Y * X)
    print(f"Max elements: {max_elements}, Chunk size: {chunk_size}")
    return max(1, chunk_size)

def mask_segmentation_preprocessing_gpu(internal_segmentation: InternalSegmentation):
    """
    Simple and efficient GPU-accelerated mask segmentation preprocessing
    """
    our_zarr_structure = open_zarr_structure_from_path(
        internal_segmentation.intermediate_zarr_structure_path
    )

    internal_segmentation.primary_descriptor = SegmentationPrimaryDescriptor.three_d_volume

    segmentation_data_gr = our_zarr_structure.create_group(
        LATTICE_SEGMENTATION_DATA_GROUPNAME
    )

    internal_segmentation.value_to_segment_id_dict = {}

    set_segmentation_custom_data(internal_segmentation, our_zarr_structure)

    if "segmentation_ids_mapping" not in internal_segmentation.custom_data:
        internal_segmentation.custom_data["segmentation_ids_mapping"] = {
            s.stem: s.stem for s in internal_segmentation.segmentation_input_path
        }

    segmentation_ids_mapping = internal_segmentation.custom_data["segmentation_ids_mapping"]
    processed_count = 0
    
    for mask in internal_segmentation.segmentation_input_path:
        try:
            with mrcfile.mmap(str(mask.resolve()), mode='r+', permissive=True) as mrc_original:
                if mrc_original.data is not None:
                    mrc_original.update_header_from_data()
                else:
                    print("Data is None; cannot update header.")
                
                shape = mrc_original.data.shape
                header = mrc_original.header

                current_order = (
                    int(header.mapc) - 1, 
                    int(header.mapr) - 1, 
                    int(header.maps) - 1
                )

                chunk_axis = 0
                chunk_total = shape[chunk_axis]
                t = shape
                shape = t[::-1]
                unique_values_set = set()
                chunk_data_cpu = np.empty(shape, dtype=mrc_original.data.dtype)
                chunk_size = get_optimal_chunk_size(shape[0], shape[1], element_size=8, reserve=0.5)

                for chunk_start in range(0, chunk_total, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, chunk_total)
                    data_chunk = mrc_original.data[chunk_start:chunk_end, :, :]
                    print(f"Processing chunk {chunk_start} to {chunk_end}")
                    data_gpu = cp.asarray(data_chunk)
                    data_gpu = gpu_normalize_axis_order(data_gpu, current_order)

                    if data_gpu.dtype.kind == 'f':
                        data_gpu = data_gpu.astype("i4")

                    unique_values = cp.unique(data_gpu)
                    unique_values_set.update(cp.asnumpy(unique_values))

                    chunk_data_cpu[ :, :, chunk_start:chunk_end] = cp.asnumpy(data_gpu)

                    del data_gpu
                    cp.get_default_memory_pool().free_all_blocks()

                value_to_segment_id_dict = {
                    int(value): int(value) for value in unique_values_set
                }

                lattice_id = segmentation_ids_mapping[mask.stem]
                    
                internal_segmentation.value_to_segment_id_dict[lattice_id] = value_to_segment_id_dict
                internal_segmentation.map_headers[lattice_id] = header

                lattice_gr = segmentation_data_gr.create_group(lattice_id)
                store_segmentation_data_in_zarr_structure(
                    original_data=chunk_data_cpu,
                    lattice_data_group=lattice_gr,
                    value_to_segment_id_dict_for_specific_lattice_id=value_to_segment_id_dict,
                    params_for_storing=internal_segmentation.params_for_storing,
                )
                    
                processed_count += 1
                    
        except Exception as e:
            print(f"Error processing mask {mask} on GPU {gpu_id}: {e}")
            continue
    
    print(f"GPU Mask segmentation processed: {processed_count}/{len(internal_segmentation.segmentation_input_path)} masks")
    print(f"Mask headers: {internal_segmentation.map_headers}")
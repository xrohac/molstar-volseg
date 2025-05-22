import numcodecs
import numpy as np
import zarr
import time as times
import cupy as cp
from cellstar_preprocessor.flows.common import create_dataset_wrapper
from cellstar_preprocessor.flows.segmentation.downsampling_level_dict import (
    DownsamplingLevelDict,
)
from cellstar_preprocessor.flows.segmentation.segmentation_set_table import (
    SegmentationSetTable,
)
from cellstar_preprocessor.tools.magic_kernel_downsampling_3d.magic_kernel_downsampling_3d import (
    MagicKernel3dDownsampler,
)

def store_downsampling_levels_in_zarr(
    levels_list: list[DownsamplingLevelDict],
    lattice_data_group: zarr.Group,
    params_for_storing: dict,
    time_frame: str,
):
    """Store downsampling levels in zarr format."""
    for level_dict in levels_list:
        grid = level_dict.get_grid()
        table = level_dict.get_set_table()
        ratio = level_dict.get_ratio()

        new_level_group: zarr.Group = lattice_data_group.create_group(str(ratio))
        time_frame_data_group = new_level_group.create_group(time_frame)

        grid_arr = create_dataset_wrapper(
            zarr_group=time_frame_data_group,
            data=grid,
            name="grid",
            shape=grid.shape,
            dtype=grid.dtype,
            params_for_storing=params_for_storing,
        )

        table_obj_arr = time_frame_data_group.create_dataset(
            name="set_table",
            dtype=object,
            object_codec=numcodecs.JSON(),
            shape=1,
        )

        table_obj_arr[...] = [table.get_serializable_repr()]

def downsample_categorical_data_optimized(
    magic_kernel: MagicKernel3dDownsampler,
    previous_level_dict: DownsamplingLevelDict,
    current_set_table: SegmentationSetTable,
) -> DownsamplingLevelDict:
    """
    Fast implementation that optimizes the original algorithm.
    """
    previous_level_grid = previous_level_dict.get_grid()
    previous_level_set_table = previous_level_dict.get_set_table()
    
    output_shape = (
        previous_level_grid.shape[0] // 2 + previous_level_grid.shape[0] % 2,
        previous_level_grid.shape[1] // 2 + previous_level_grid.shape[1] % 2,
        previous_level_grid.shape[2] // 2 + previous_level_grid.shape[2] % 2
    )
    current_level_grid = np.full(output_shape, np.nan, dtype=previous_level_grid.dtype)
    
    start_time = times.time()
    
    target_voxels_coords = np.array(
        magic_kernel.extract_target_voxels_coords(previous_level_grid.shape)
    )
    
    id_category_cache = {}
    
    def get_cached_categories(value):
        if value not in id_category_cache:
            id_category_cache[value] = previous_level_set_table.get_categories((value,))[0]
        return id_category_cache[value]
    
    block_category_cache = {}
    
    category_id_cache = {}
    
    def get_cached_category_id(category_set):
        key = tuple(sorted(category_set))
        if key not in category_id_cache:
            category_id_cache[key] = current_set_table.resolve_category(category_set)
        return category_id_cache[key]
    
    print(f"Processing {len(target_voxels_coords)} blocks with optimized logic...")
    
    origin_coords = np.array([0, 0, 0])
    max_coords = np.array(previous_level_grid.shape) - 1
    
    processed_count = 0
    block_cache_hits = 0
    
    for start_coords in target_voxels_coords:
        output_z = start_coords[0] // 2
        output_y = start_coords[1] // 2
        output_x = start_coords[2] // 2
        
        end_coords = np.maximum(np.minimum(start_coords + 2, max_coords), origin_coords)
        
        if any((end_coords - start_coords) <= 0):
            continue
        
        block = previous_level_grid[
            start_coords[0]:end_coords[0],
            start_coords[1]:end_coords[1],
            start_coords[2]:end_coords[2]
        ]
        
        block_values = tuple(sorted(set(int(x) for x in block.flatten())))
        
        if block_values in block_category_cache:
            category_id = block_category_cache[block_values]
            block_cache_hits += 1
        else:
            if not block_values:
                union_set = set()
            else:
                categories_list = []
                for val in block_values:
                    categories_list.append(get_cached_categories(val))
                
                union_set = set().union(*categories_list)
            category_id = get_cached_category_id(union_set)
            
            block_category_cache[block_values] = category_id
        
        current_level_grid[output_z, output_y, output_x] = category_id
        processed_count += 1
    assert (
        np.isnan(current_level_grid).any() == False
    ), f"Segmentation grid contain NAN values"
    
    end_time = times.time()
    print(f"Time taken for fast implementation: {end_time - start_time:.4f} seconds")
    print(f"Processed {processed_count} blocks with {len(block_category_cache)} unique patterns")
    print(f"Block cache hits: {block_cache_hits} ({(block_cache_hits / processed_count * 100):.2f}% hit rate)")
    
    new_dict = DownsamplingLevelDict(
        {
            "ratio": round(previous_level_dict.get_ratio() * 2),
            "grid": current_level_grid,
            "set_table": current_set_table,
        }
    )
    
    return new_dict

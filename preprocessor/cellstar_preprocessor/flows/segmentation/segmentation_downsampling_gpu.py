import math
import cupy as cp
import numpy as np
import zarr

from cellstar_preprocessor.flows.common import (
    compute_downsamplings_to_be_stored,
    compute_number_of_downsampling_steps,
    open_zarr_structure_from_path,
)
from cellstar_preprocessor.flows.constants import (
    LATTICE_SEGMENTATION_DATA_GROUPNAME,
    MESH_SEGMENTATION_DATA_GROUPNAME,
    MESH_VERTEX_DENSITY_THRESHOLD,
    MIN_GRID_SIZE,
)
from cellstar_preprocessor.flows.segmentation.category_set_downsampling_methods import (
    store_downsampling_levels_in_zarr,
)
from cellstar_preprocessor.flows.segmentation.downsampling_level_dict import (
    DownsamplingLevelDict,
)
from cellstar_preprocessor.flows.segmentation.helper_methods import (
    compute_vertex_density,
    simplify_meshes,
    store_mesh_data_in_zarr,
)
from cellstar_preprocessor.flows.segmentation.segmentation_set_table import (
    SegmentationSetTable,
)
from cellstar_preprocessor.model.input import SegmentationPrimaryDescriptor
from cellstar_preprocessor.model.segmentation import InternalSegmentation
from cellstar_preprocessor.tools.magic_kernel_downsampling_3d.magic_kernel_downsampling_3d import (
    MagicKernel3dDownsampler,
)

from cellstar_preprocessor.flows.volume.helper_methods import (
    gaussian_kernel_3d,
)

def sff_segmentation_downsampling_gpu(internal_segmentation: InternalSegmentation):
    zarr_structure = open_zarr_structure_from_path(
        internal_segmentation.intermediate_zarr_structure_path
    )

    if internal_segmentation.primary_descriptor == SegmentationPrimaryDescriptor.three_d_volume:
        lat_group = zarr_structure[LATTICE_SEGMENTATION_DATA_GROUPNAME]
        for lattice_id, lattice_gr in lat_group.groups():
            orig_res_gr: zarr.Group = lattice_gr["1"]
            for time_frame, time_gr in orig_res_gr.groups():
                orig_np = orig_res_gr[time_frame].grid
                orig_gpu = cp.asarray(orig_np)

                steps = compute_number_of_downsampling_steps(
                    int_vol_or_seg=internal_segmentation,
                    min_grid_size=MIN_GRID_SIZE,
                    input_grid_size=math.prod(orig_np.shape),
                    force_dtype=orig_np.dtype,
                    factor=2**3,
                )
                ratios = compute_downsamplings_to_be_stored(
                    int_vol_or_seg=internal_segmentation,
                    number_of_downsampling_steps=steps,
                    input_grid_size=math.prod(orig_np.shape),
                    dtype=orig_np.dtype,
                    factor=2**3,
                )

                _create_category_set_downsamplings_gpu(
                    original_gpu=orig_gpu,
                    downsampling_steps=steps,
                    ratios_to_be_stored=ratios,
                    data_group=lattice_gr,
                    value_to_seg_id=internal_segmentation.value_to_segment_id_dict[lattice_id],
                    params_for_storing=internal_segmentation.params_for_storing,
                    time_frame=time_frame,
                )

            if internal_segmentation.downsampling_parameters.remove_original_resolution:
                del lattice_gr["1"]
                print("Original resolution data removed for segmentation")

    elif internal_segmentation.primary_descriptor == SegmentationPrimaryDescriptor.mesh_list:
        simplification_curve = internal_segmentation.simplification_curve
        calc_mode = "area"
        density_threshold = MESH_VERTEX_DENSITY_THRESHOLD[calc_mode]

        segm_data_gr = zarr_structure[MESH_SEGMENTATION_DATA_GROUPNAME]
        for set_id, set_gr in segm_data_gr.groups():
            for tf_idx, tf_gr in set_gr.groups():
                for seg_id, seg_gr in tf_gr.groups():
                    base_mesh_group = seg_gr["1"]

                    for level, fraction in simplification_curve.items():
                        if density_threshold and compute_vertex_density(base_mesh_group, mode=calc_mode) <= density_threshold:
                            break
                        if fraction == 1:
                            continue

                        mesh_dict = simplify_meshes(
                            base_mesh_group,
                            ratio=fraction,
                            segment_id=seg_id,
                        )
                        mesh_dict = {mid: m for mid, m in mesh_dict.items()
                                     if m["attrs"]["num_vertices"] > 0}
                        if not mesh_dict:
                            break

                        base_mesh_group = store_mesh_data_in_zarr(
                            mesh_dict,
                            seg_gr,
                            detail_level=level,
                            params_for_storing=internal_segmentation.params_for_storing,
                        )

                    if internal_segmentation.downsampling_parameters.remove_original_resolution:
                        del seg_gr["1"]

        if internal_segmentation.downsampling_parameters.remove_original_resolution:
            internal_segmentation.simplification_curve.pop(1, None)

    print("Segmentation downsampled")


def _create_category_set_downsamplings_gpu(
    original_gpu: cp.ndarray,
    downsampling_steps: int,
    ratios_to_be_stored: list[int],
    data_group: zarr.Group,
    value_to_seg_id: dict[int, int],
    params_for_storing: dict,
    time_frame: int,
):
    """
    GPU-accelerated version of _create_category_set_downsamplings.
    original_gpu: a cupy.ndarray of the segmentation grid.
    """
    levels = [
        DownsamplingLevelDict({
            "ratio": 1,
            "grid": cp.asnumpy(original_gpu),
            "set_table": SegmentationSetTable(
                cp.asnumpy(original_gpu).astype(int),
                value_to_seg_id
            )
        })
    ]

    for i in range(downsampling_steps):
        prev = levels[i]
        down_gpu = _gpu_max_pool(prev.get_grid(), factor=2)
        ratio = round(prev.get_ratio() * 2)

        down_np = cp.asnumpy(down_gpu).astype(int)
        set_table = SegmentationSetTable(down_np, value_to_seg_id)

        levels.append(
            DownsamplingLevelDict({
                "ratio": ratio,
                "grid": down_np,
                "set_table": set_table,
            })
        )
        
        del down_gpu
        del down_np
        cp.get_default_memory_pool().free_all_blocks()

    levels.pop(0)

    levels = [lvl for lvl in levels if lvl.get_ratio() in ratios_to_be_stored]

    store_downsampling_levels_in_zarr(
        levels,
        lattice_data_group=data_group,
        params_for_storing=params_for_storing,
        time_frame=time_frame,
    )

    print(f"Stored GPU downsampling levels: {[lvl.get_ratio() for lvl in levels]}")

def _gpu_max_pool(arr: cp.ndarray, factor: int) -> cp.ndarray:
    """
    Simple 2× downsample (factor=2) by cube max-pooling on GPU.
    For general factor, each axis is reduced by that factor.
    """
    assert arr.ndim == 3, "Expect 3D volume"
    f = factor
    sx, sy, sz = arr.shape
    nx, ny, nz = sx // f, sy // f, sz // f
    arr = arr[: nx * f, : ny * f, : nz * f]
    reshaped = arr.reshape((nx, f, ny, f, nz, f))
    return reshaped.max(axis=(1, 3, 5))

# import math
# import cupy as cp
# import zarr
# from cupyx.scipy.ndimage import convolve as gpu_convolve

# from cellstar_preprocessor.flows.common import (
#     compute_downsamplings_to_be_stored,
#     compute_number_of_downsampling_steps,
#     open_zarr_structure_from_path,
# )
# from cellstar_preprocessor.flows.constants import (
#     LATTICE_SEGMENTATION_DATA_GROUPNAME,
#     MESH_SEGMENTATION_DATA_GROUPNAME,
#     MESH_VERTEX_DENSITY_THRESHOLD,
#     MIN_GRID_SIZE,
# )
# from cellstar_preprocessor.flows.segmentation.category_set_downsampling_methods import (
#     store_downsampling_levels_in_zarr,
# )
# from cellstar_preprocessor.flows.segmentation.downsampling_level_dict import (
#     DownsamplingLevelDict,
# )
# from cellstar_preprocessor.flows.segmentation.helper_methods import (
#     compute_vertex_density,
#     simplify_meshes,
#     store_mesh_data_in_zarr,
# )
# from cellstar_preprocessor.flows.segmentation.segmentation_set_table import (
#     SegmentationSetTable,
# )
# from cellstar_preprocessor.model.input import SegmentationPrimaryDescriptor
# from cellstar_preprocessor.model.segmentation import InternalSegmentation
# from cellstar_preprocessor.tools.magic_kernel_downsampling_3d.magic_kernel_downsampling_3d import (
#     MagicKernel3dDownsampler,
# )

# from cellstar_preprocessor.flows.volume.helper_methods import (
#     gaussian_kernel_3d,
# )

# def get_optimal_chunk_size(Y, X, element_size=4, reserve=0.7):
#     free_mem = cp.cuda.Device(0).mem_info[0]
#     usable_mem = int(free_mem * reserve)
#     print(f"Free memory: {free_mem}, Usable memory: {usable_mem}")
#     max_elements = usable_mem // element_size
#     chunk_size = max_elements // (Y * X)
#     print(f"Max elements: {max_elements}, Chunk size: {chunk_size}")
#     return max(1, chunk_size)

# def sff_segmentation_downsampling_gpu(internal_segmentation):
#     zarr_structure = open_zarr_structure_from_path(
#         internal_segmentation.intermediate_zarr_structure_path
#     )

#     if internal_segmentation.primary_descriptor == SegmentationPrimaryDescriptor.three_d_volume:
#         lat_group = zarr_structure[LATTICE_SEGMENTATION_DATA_GROUPNAME]
#         for lattice_id, lattice_gr in lat_group.groups():
#             orig_res_gr = lattice_gr["1"]
#             for time_frame, time_gr in orig_res_gr.groups():
#                 orig_np = orig_res_gr[time_frame].grid
#                 Y, X, Z = orig_np.shape

#                 chunk_depth = get_optimal_chunk_size(Y, X, element_size=orig_np.dtype.itemsize, reserve=0.7)
#                 print(f"Optimal chunk size (Z): {chunk_depth}")

#                 steps = compute_number_of_downsampling_steps(
#                     int_vol_or_seg=internal_segmentation,
#                     min_grid_size=MIN_GRID_SIZE,
#                     input_grid_size=math.prod(orig_np.shape),
#                     force_dtype=orig_np.dtype,
#                     factor=2**3,
#                 )
#                 ratios = compute_downsamplings_to_be_stored(
#                     int_vol_or_seg=internal_segmentation,
#                     number_of_downsampling_steps=steps,
#                     input_grid_size=math.prod(orig_np.shape),
#                     dtype=orig_np.dtype,
#                     factor=2**3,
#                 )

#                 chunked_levels = {ratio: [] for ratio in ratios}

#                 for z_start in range(0, Z, chunk_depth):
#                     z_end = min(z_start + chunk_depth, Z)
#                     chunk_np = orig_np[:, :, z_start:z_end]
#                     chunk_gpu = cp.asarray(chunk_np)

#                     levels = [
#                         DownsamplingLevelDict({
#                             "ratio": 1,
#                             "grid": cp.asnumpy(chunk_gpu),
#                             "set_table": SegmentationSetTable(
#                                 cp.asnumpy(chunk_gpu).astype(int),
#                                 internal_segmentation.value_to_segment_id_dict[lattice_id]
#                             )
#                         })
#                     ]
#                     for i in range(steps):
#                         prev = levels[i]
#                         down_gpu = _gpu_max_pool(prev.get_grid(), factor=2)
#                         down_gpu = cp.transpose(down_gpu, (2, 1, 0))
#                         ratio = round(prev.get_ratio() * 2)
#                         down_np = cp.asnumpy(down_gpu).astype(int)
#                         set_table = SegmentationSetTable(down_np, internal_segmentation.value_to_segment_id_dict[lattice_id])
#                         levels.append(
#                             DownsamplingLevelDict({
#                                 "ratio": ratio,
#                                 "grid": down_np,
#                                 "set_table": set_table,
#                             })
#                         )
#                         del down_gpu
#                         del down_np
#                         cp.get_default_memory_pool().free_all_blocks()
#                     levels.pop(0)
#                     levels = [lvl for lvl in levels if lvl.get_ratio() in ratios]

#                     for lvl in levels:
#                         chunked_levels[lvl.get_ratio()].append(lvl.get_grid())

#                     del chunk_gpu
#                     cp.get_default_memory_pool().free_all_blocks()

#                 final_levels = []
#                 for ratio in ratios:
#                     if chunked_levels[ratio]:
#                         combined_grid = np.concatenate(chunked_levels[ratio], axis=2)
#                         set_table = SegmentationSetTable(combined_grid.astype(int), internal_segmentation.value_to_segment_id_dict[lattice_id])
#                         final_levels.append(
#                             DownsamplingLevelDict({
#                                 "ratio": ratio,
#                                 "grid": combined_grid,
#                                 "set_table": set_table,
#                             })
#                         )
#                 store_downsampling_levels_in_zarr(
#                     final_levels,
#                     lattice_data_group=lattice_gr,
#                     params_for_storing=internal_segmentation.params_for_storing,
#                     time_frame=time_frame,
#                 )

#             if internal_segmentation.downsampling_parameters.remove_original_resolution:
#                 del lattice_gr["1"]
#                 print("Original resolution data removed for segmentation")

#     elif internal_segmentation.primary_descriptor == SegmentationPrimaryDescriptor.mesh_list:
#         simplification_curve = internal_segmentation.simplification_curve
#         calc_mode = "area"
#         density_threshold = MESH_VERTEX_DENSITY_THRESHOLD[calc_mode]

#         segm_data_gr = zarr_structure[MESH_SEGMENTATION_DATA_GROUPNAME]
#         for set_id, set_gr in segm_data_gr.groups():
#             for tf_idx, tf_gr in set_gr.groups():
#                 for seg_id, seg_gr in tf_gr.groups():
#                     base_mesh_group = seg_gr["1"]

#                     for level, fraction in simplification_curve.items():
#                         if density_threshold and compute_vertex_density(base_mesh_group, mode=calc_mode) <= density_threshold:
#                             break
#                         if fraction == 1:
#                             continue

#                         mesh_dict = simplify_meshes(
#                             base_mesh_group,
#                             ratio=fraction,
#                             segment_id=seg_id,
#                         )
#                         mesh_dict = {mid: m for mid, m in mesh_dict.items()
#                                      if m["attrs"]["num_vertices"] > 0}
#                         if not mesh_dict:
#                             break

#                         base_mesh_group = store_mesh_data_in_zarr(
#                             mesh_dict,
#                             seg_gr,
#                             detail_level=level,
#                             params_for_storing=internal_segmentation.params_for_storing,
#                         )

#                     if internal_segmentation.downsampling_parameters.remove_original_resolution:
#                         del seg_gr["1"]

#         if internal_segmentation.downsampling_parameters.remove_original_resolution:
#             internal_segmentation.simplification_curve.pop(1, None)

#     print("Segmentation downsampled")
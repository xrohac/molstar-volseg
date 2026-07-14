import mrcfile
import numpy as np

from cellstar_preprocessor.flows.common import (
    open_zarr_structure_from_path,
    set_segmentation_custom_data,
)
from cellstar_preprocessor.flows.constants import LATTICE_SEGMENTATION_DATA_GROUPNAME
from cellstar_preprocessor.flows.gpu_backend import (
    TORCH_AVAILABLE,
    backend_name,
    get_device,
    gpu_is_available,
    to_device_safe,
    to_numpy,
    torch,
)
from cellstar_preprocessor.flows.segmentation.helper_methods import (
    store_segmentation_data_in_zarr_structure,
)
from cellstar_preprocessor.flows.segmentation.segmentation_set_table import (
    unique_int_values,
)
# CPU reference used as a transparent fallback when no usable GPU is present.
from cellstar_preprocessor.flows.segmentation.mask_segmentation_preprocessing import (
    mask_segmentation_preprocessing,
)
from cellstar_preprocessor.model.input import SegmentationPrimaryDescriptor
from cellstar_preprocessor.model.segmentation import InternalSegmentation


def gpu_normalize_axis_order(arr: "torch.Tensor", current_order: tuple) -> "torch.Tensor":
    """
    Normalize axis order to (0, 1, 2) using the active PyTorch device.
    Mirrors the original CuPy behaviour exactly.
    """
    if current_order != (0, 1, 2):
        print(f"Reordering axes from {current_order}...")
        ao = {v: i for i, v in enumerate(current_order)}
        return arr.permute(ao[2], ao[1], ao[0])
    return arr.permute(*reversed(range(arr.ndim)))


def _free_gpu_bytes(device) -> int | None:
    """Free device memory in bytes, or None when it cannot be queried."""
    try:
        if torch is not None and device is not None and getattr(device, "type", "") == "cuda":
            free, _total = torch.cuda.mem_get_info()
            return int(free)
    except Exception:
        pass
    return None


def get_optimal_chunk_size(Y, X, element_size=4, reserve=0.5, device=None):
    """
    Chunk height (along the streaming axis) that fits in the memory budget.

    On CUDA/ROCm we size against real free VRAM; other backends (DirectML, MPS)
    don't expose a portable query, so we use a conservative fixed budget.
    """
    free = _free_gpu_bytes(device)
    if free is None:
        budget = 512 * 1024 * 1024  # 512 MB working set
    else:
        budget = int(free * reserve)
    max_elements = budget // element_size
    chunk_size = max_elements // max(1, (Y * X))
    print(f"Chunk budget: {budget} bytes, chunk size: {max(1, chunk_size)}")
    return max(1, chunk_size)


def mask_segmentation_preprocessing_gpu(internal_segmentation: InternalSegmentation):
    """
    Vendor-neutral GPU mask segmentation preprocessing (AMD/NVIDIA/Intel/Apple)
    via PyTorch, with a transparent CPU fallback.
    """
    if not (TORCH_AVAILABLE and gpu_is_available()):
        print("[mask] no usable GPU backend - falling back to CPU preprocessing")
        return mask_segmentation_preprocessing(internal_segmentation)

    try:
        _mask_segmentation_preprocessing_torch(internal_segmentation)
    except Exception as e:  # pragma: no cover - hardware/driver dependent
        print(f"[mask] GPU preprocessing failed ({e!r}); falling back to CPU")
        return mask_segmentation_preprocessing(internal_segmentation)


def _mask_segmentation_preprocessing_torch(internal_segmentation: InternalSegmentation):
    device = get_device()
    print(f"[mask] preprocessing on '{backend_name()}' backend")

    our_zarr_structure = open_zarr_structure_from_path(
        internal_segmentation.intermediate_zarr_structure_path
    )

    internal_segmentation.primary_descriptor = (
        SegmentationPrimaryDescriptor.three_d_volume
    )

    segmentation_data_gr = our_zarr_structure.create_group(
        LATTICE_SEGMENTATION_DATA_GROUPNAME
    )

    internal_segmentation.value_to_segment_id_dict = {}

    set_segmentation_custom_data(internal_segmentation, our_zarr_structure)

    if "segmentation_ids_mapping" not in internal_segmentation.custom_data:
        internal_segmentation.custom_data["segmentation_ids_mapping"] = {
            s.stem: s.stem for s in internal_segmentation.segmentation_input_path
        }

    segmentation_ids_mapping = internal_segmentation.custom_data[
        "segmentation_ids_mapping"
    ]
    processed_count = 0

    for mask in internal_segmentation.segmentation_input_path:
        try:
            with mrcfile.mmap(
                str(mask.resolve()), mode="r+", permissive=True
            ) as mrc_original:
                if mrc_original.data is not None:
                    mrc_original.update_header_from_data()
                else:
                    print("Data is None; cannot update header.")

                shape = mrc_original.data.shape
                header = mrc_original.header

                current_order = (
                    int(header.mapc) - 1,
                    int(header.mapr) - 1,
                    int(header.maps) - 1,
                )

                chunk_axis = 0
                chunk_total = shape[chunk_axis]
                shape = shape[::-1]
                unique_values_set = set()
                # Float masks are stored as integer labels (matches the CPU
                # reference's astype('i4')). Storing them as float would make the
                # segment ids floats (e.g. "0.0"), which breaks serialization
                # downstream (SegmentSetTable expects integer id strings).
                out_dtype = (
                    np.int32
                    if np.issubdtype(mrc_original.data.dtype, np.floating)
                    else mrc_original.data.dtype
                )
                chunk_data_cpu = np.empty(shape, dtype=out_dtype)
                chunk_size = get_optimal_chunk_size(
                    shape[0], shape[1], element_size=8, reserve=0.5, device=device
                )

                for chunk_start in range(0, chunk_total, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, chunk_total)
                    data_chunk = mrc_original.data[chunk_start:chunk_end, :, :]
                    print(f"Processing chunk {chunk_start} to {chunk_end}")

                    # to_device_safe normalizes unsigned/odd dtypes (which fatally
                    # abort DirectML) to int64/float32 before the device transfer.
                    data_gpu = to_device_safe(data_chunk)
                    data_gpu = gpu_normalize_axis_order(data_gpu, current_order)

                    chunk_np = to_numpy(data_gpu.contiguous())
                    # float masks become int labels (matches the CPU reference's
                    # 'i4' cast); done on the host to avoid device-side int casts.
                    if np.issubdtype(chunk_np.dtype, np.floating):
                        chunk_np = chunk_np.astype(np.int32)
                    # collect label values on the host copy we already need.
                    # bincount-based detection is ~3x faster than np.unique's sort
                    # on these large label chunks (see unique_int_values).
                    unique_values_set.update(unique_int_values(chunk_np).tolist())

                    chunk_data_cpu[:, :, chunk_start:chunk_end] = chunk_np

                    del data_gpu

                value_to_segment_id_dict = {
                    int(value): int(value) for value in unique_values_set
                }

                lattice_id = segmentation_ids_mapping[mask.stem]

                internal_segmentation.value_to_segment_id_dict[lattice_id] = (
                    value_to_segment_id_dict
                )
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
            print(f"Error processing mask {mask} on GPU: {e}")
            continue

    print(
        f"GPU Mask segmentation processed: {processed_count}/"
        f"{len(internal_segmentation.segmentation_input_path)} masks"
    )
    print(f"Mask headers: {internal_segmentation.map_headers}")

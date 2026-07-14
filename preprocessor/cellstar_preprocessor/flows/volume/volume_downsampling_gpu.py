import math
import time as times

import dask.array as da
import numpy as np
import zarr

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
from cellstar_preprocessor.flows.gpu_backend import (
    TORCH_AVAILABLE,
    backend_name,
    conv2d_supported,
    get_device,
    to_numpy,
    torch,
)
from cellstar_preprocessor.flows.volume.helper_methods import (
    generate_kernel_3d_arr,
    store_volume_data_in_zarr_stucture,
)
# CPU reference - used as a transparent fallback when no usable GPU is present.
from cellstar_preprocessor.flows.volume.volume_downsampling import volume_downsampling
from cellstar_preprocessor.model.volume import InternalVolume


def volume_downsampling_gpu(internal_volume: InternalVolume):
    """
    GPU volume downsampling that runs on any vendor (AMD/NVIDIA/Intel/Apple)
    through PyTorch, with a transparent CPU fallback.

    If no usable GPU backend is available we defer to the CPU reference
    (``volume_downsampling``) so the pipeline never breaks on machines without
    the optional accelerator libraries installed.
    """
    if not (TORCH_AVAILABLE and conv2d_supported()):
        print("[volume] no usable GPU backend - falling back to CPU downsampling")
        return volume_downsampling(internal_volume)

    try:
        _volume_downsampling_torch(internal_volume)
    except Exception as e:  # pragma: no cover - hardware/driver dependent
        print(f"[volume] GPU downsampling failed ({e!r}); falling back to CPU")
        return volume_downsampling(internal_volume)


# peak GPU working set for one Z-tile's input slab (kept small for limited VRAM)
_VOLUME_TILE_BUDGET_BYTES = 256 * 1024 * 1024


def _reflect_pad_hw(t: "torch.Tensor", p: int) -> "torch.Tensor":
    """
    Reflect-pad only the H,W axes (1,2) of a (D,H,W) tensor by ``p``
    (whole-sample symmetric == numpy 'reflect' == scipy 'mirror' == torch
    'reflect'). Built from index_select + cat because DirectML has no N-D
    reflection padding but does support gather/concat. Z is padded on the host.
    """
    for axis in (1, 2):
        n = t.shape[axis]
        left = t.index_select(axis, torch.arange(p, 0, -1, device=t.device))
        right = t.index_select(axis, torch.arange(n - 2, n - 2 - p, -1, device=t.device))
        t = torch.cat([left, t, right], dim=axis)
    return t


def _tiled_magic_downsample(
    vol_np: np.ndarray, weight_planes: "torch.Tensor", pad: int, device
) -> np.ndarray:
    """
    Memory-bounded GPU downsampling of a host volume, for arbitrarily large data.

    Fuses "convolve (mirror) + take every 2nd voxel" for the 5x5x5 magic kernel
    using conv2d (so it runs on GPUs without 3D conv, e.g. AMD/Intel via
    DirectML). The volume never fully resides on the GPU: it is reflect-padded in
    Z on the host, then processed in tiles along the output-Z axis. Each tile
    moves only its input slab (2*tile + 2*pad slices) to the GPU, pads H,W there,
    and accumulates the ``2*pad+1`` depth planes. Peak GPU memory is bounded by
    ``_VOLUME_TILE_BUDGET_BYTES`` regardless of volume size.

    Bit-for-bit equivalent to the untiled strided conv, i.e. identical (to float
    precision) to ``convolve(vol, kernel, mode='mirror')[::2, ::2, ::2]``.
    """
    D, H, W = vol_np.shape
    out_D = (D + 1) // 2
    ksize = weight_planes.shape[1]  # 2*pad + 1

    # Z reflect-pad on the host (cheap: +2*pad planes). numpy 'reflect' == mirror.
    padded_z = np.pad(vol_np, ((pad, pad), (0, 0), (0, 0)), mode="reflect")

    # tile size along output-Z so one slab stays within the GPU budget
    plane_bytes = (H + 2 * pad) * (W + 2 * pad) * 4
    tile = max(1, (_VOLUME_TILE_BUDGET_BYTES // max(1, plane_bytes) - 2 * pad) // 2)

    out_parts = []
    for zo0 in range(0, out_D, tile):
        zo1 = min(zo0 + tile, out_D)
        t = zo1 - zo0
        base = 2 * zo0
        # padded-Z slice covering outputs [zo0, zo1): local index (2*j + dz)
        slab = padded_z[base:base + 2 * (t - 1) + 2 * pad + 1]
        slab_t = torch.as_tensor(np.ascontiguousarray(slab)).to(
            device=device, dtype=torch.float32
        )
        slab_t = _reflect_pad_hw(slab_t, pad)  # (slab_d, Hp, Wp)
        acc = None
        for dz in range(ksize):
            chan = slab_t[dz:dz + 2 * t:2].unsqueeze(1).contiguous()  # (t,1,Hp,Wp)
            conv = torch.nn.functional.conv2d(
                chan, weight_planes[:, dz:dz + 1], stride=2
            )  # (t,1,H',W')
            acc = conv if acc is None else acc + conv
        out_parts.append(to_numpy(acc[:, 0]))  # (t, H', W')
        del slab_t, acc
    return np.concatenate(out_parts, axis=0)


def _decode_quantized_numpy(data_dict: dict) -> np.ndarray:
    """Host (numpy) reverse of the log-quantization - avoids decoding a huge
    volume on the GPU. Elementwise and cheap."""
    delta = (data_dict["max"] - data_dict["min"]) / (data_dict["num_steps"] - 1)
    log = data_dict["data"].astype(np.float32)
    log = log * float(delta) + float(data_dict["min"])
    original = np.exp(log) - 1.0 + float(data_dict["to_remove_negatives"])
    return original.astype(np.float32)


def _volume_downsampling_torch(internal_volume: InternalVolume):
    device = get_device()
    print(f"[volume] downsampling on '{backend_name()}' backend")

    zarr_structure = open_zarr_structure_from_path(
        internal_volume.intermediate_zarr_structure_path
    )

    # Magic (1,4,6,4,1) kernel, identical to the CPU reference. It is NOT
    # separable (it is defined on Chebyshev-distance shells). It is symmetric, so
    # cross-correlation (conv2d) equals true convolution. We reshape it to
    # (1, ksize, ksize, ksize): the leading dim is the conv2d output channel and
    # the second dim (kernel Z-planes) becomes the conv2d input channels.
    kernel_np = generate_kernel_3d_arr(list(DOWNSAMPLING_KERNEL)).astype(np.float32)
    pad = kernel_np.shape[0] // 2  # == 2
    weight_planes = torch.as_tensor(kernel_np).to(device).reshape(1, *kernel_np.shape)

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
                data_dict["data"] = original_data_arr[:]
                current_np = _decode_quantized_numpy(data_dict)
                # sizing must use the DECODED dtype (matches the CPU reference),
                # not the stored quantized dtype.
                sizing_dtype = np.dtype(data_dict["src_type"])
            else:
                # host float32; the volume is streamed to the GPU in tiles so it
                # never needs to fit in VRAM whole.
                current_np = np.ascontiguousarray(original_data_arr[:]).astype(
                    np.float32, copy=False
                )
                sizing_dtype = original_data_arr.dtype

            input_grid_size = math.prod(current_np.shape)

            downsampling_steps = compute_number_of_downsampling_steps(
                int_vol_or_seg=internal_volume,
                min_grid_size=MIN_GRID_SIZE,
                input_grid_size=input_grid_size,
                factor=2**3,
                force_dtype=sizing_dtype,
            )
            ratios_to_be_stored = compute_downsamplings_to_be_stored(
                int_vol_or_seg=internal_volume,
                number_of_downsampling_steps=downsampling_steps,
                input_grid_size=input_grid_size,
                factor=2**3,
                dtype=sizing_dtype,
            )

            start_time = times.time()
            # ``current_np`` is the host volume; each level is downsampled on the
            # GPU in Z-tiles (see _tiled_magic_downsample) and returned to the
            # host, so peak VRAM is bounded no matter how large the volume is.
            for i in range(downsampling_steps):
                current_ratio = 2 ** (i + 1)

                # OPTIMIZATION: fuse "convolve (mirror) + take every 2nd voxel"
                # into strided conv2d. Identical to the original
                # (convolve -> [::2, ::2, ::2]) while computing only 1/8 of the
                # outputs, and runs on GPUs that lack 3D conv (AMD/Intel/DirectML).
                current_np = _tiled_magic_downsample(current_np, weight_planes, pad, device)

                if current_ratio in ratios_to_be_stored:
                    store_volume_data_in_zarr_stucture(
                        data=da.from_array(current_np),
                        volume_data_group=zarr_structure[VOLUME_DATA_GROUPNAME],
                        params_for_storing=internal_volume.params_for_storing,
                        force_dtype=internal_volume.volume_force_dtype,
                        resolution=current_ratio,
                        time_frame=time,
                        channel=channel_id,
                    )

            elapsed_time = times.time() - start_time
            print(f"Volume downsampled (channel {channel_id}, timeframe {time}) "
                  f"in {elapsed_time:.6f} s")

    # # NOTE: remove original level resolution data
    # if internal_volume.downsampling_parameters.remove_original_resolution:
    #     del zarr_structure[VOLUME_DATA_GROUPNAME]["1"]
    #     print("Original resolution data removed")

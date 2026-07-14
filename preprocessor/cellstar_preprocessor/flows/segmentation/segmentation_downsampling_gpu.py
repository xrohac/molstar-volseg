import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import zarr

from cellstar_preprocessor.flows.common import (
    compute_downsamplings_to_be_stored,
    compute_number_of_downsampling_steps,
    open_zarr_structure_from_path,
    robust_delitem,
)
from cellstar_preprocessor.flows.constants import (
    LATTICE_SEGMENTATION_DATA_GROUPNAME,
    MESH_SEGMENTATION_DATA_GROUPNAME,
    MESH_VERTEX_DENSITY_THRESHOLD,
    MIN_GRID_SIZE,
)
# Vendor-neutral GPU backend (AMD/NVIDIA/Intel/Apple via PyTorch). The categorical
# set logic that assigns ids is inherently sequential and stays on the CPU so the
# produced grids/set-tables are byte-identical to the reference; only the bulk
# per-voxel neighbour gather + sort is offloaded to the GPU when available.
from cellstar_preprocessor.flows.gpu_backend import (
    TORCH_AVAILABLE,
    backend_name,
    get_device,
    to_numpy,
    torch,
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


class _DownsamplingStub:
    """Minimal picklable stand-in exposing what the compute_* helpers read."""

    def __init__(self, downsampling_parameters):
        self.downsampling_parameters = downsampling_parameters


def _seg_worker_count(n_lattices: int) -> int:
    """
    Number of CPU worker processes for parallel lattice downsampling.

    Override with the ``VS_MASK_WORKERS`` env var. Default is conservative (2) -
    each worker peaks at several GB for a large grid, so on a ~16 GB box more than
    2 risks swapping; bump it on machines with more RAM.
    """
    env = os.environ.get("VS_MASK_WORKERS")
    if env:
        try:
            n = int(env)
        except ValueError:
            n = 1
    else:
        n = 2
    return max(1, min(n, n_lattices))


def _downsample_one_lattice(
    zarr_path, lattice_id, value_to_seg_id, downsampling_parameters,
    params_for_storing, remove_original,
):
    """Downsample a single lattice end-to-end, writing to its own zarr subgroup.

    Idempotent: if the lattice already has downsampled levels (e.g. from a
    partially-completed parallel run) it is skipped, so a sequential retry is safe.
    """
    zs = open_zarr_structure_from_path(zarr_path)
    lattice_gr = zs[LATTICE_SEGMENTATION_DATA_GROUPNAME][lattice_id]

    already = [k for k in lattice_gr.group_keys() if k != "1"]
    if already:
        return lattice_id

    orig_res_gr = lattice_gr["1"]
    stub = _DownsamplingStub(downsampling_parameters)
    for time_frame, _tg in orig_res_gr.groups():
        orig_np = orig_res_gr[time_frame].grid[...]
        steps = compute_number_of_downsampling_steps(
            int_vol_or_seg=stub,
            min_grid_size=MIN_GRID_SIZE,
            input_grid_size=math.prod(orig_np.shape),
            force_dtype=orig_np.dtype,
            factor=2**3,
        )
        ratios = compute_downsamplings_to_be_stored(
            int_vol_or_seg=stub,
            number_of_downsampling_steps=steps,
            input_grid_size=math.prod(orig_np.shape),
            dtype=orig_np.dtype,
            factor=2**3,
        )
        _create_category_set_downsamplings_gpu(
            original_data=orig_np,
            downsampling_steps=steps,
            ratios_to_be_stored=ratios,
            data_group=lattice_gr,
            value_to_seg_id=value_to_seg_id,
            params_for_storing=params_for_storing,
            time_frame=time_frame,
        )

    if remove_original:
        robust_delitem(lattice_gr, "1")
    return lattice_id


def _lattice_worker(job):
    """Process-pool entry point: force CPU-only, then downsample one lattice."""
    os.environ["VS_DISABLE_GPU"] = "1"
    return _downsample_one_lattice(*job)


def sff_segmentation_downsampling_gpu(internal_segmentation: InternalSegmentation):
    zarr_structure = open_zarr_structure_from_path(
        internal_segmentation.intermediate_zarr_structure_path
    )

    if internal_segmentation.primary_descriptor == SegmentationPrimaryDescriptor.three_d_volume:
        lat_group = zarr_structure[LATTICE_SEGMENTATION_DATA_GROUPNAME]
        lattice_ids = list(lat_group.group_keys())
        zarr_path = str(internal_segmentation.intermediate_zarr_structure_path)
        dp = internal_segmentation.downsampling_parameters
        pfs = internal_segmentation.params_for_storing
        v2s = internal_segmentation.value_to_segment_id_dict
        remove_original = dp.remove_original_resolution

        jobs = [(zarr_path, lid, v2s[lid], dp, pfs, remove_original) for lid in lattice_ids]
        workers = _seg_worker_count(len(lattice_ids))

        if workers > 1 and len(jobs) > 1:
            # Each lattice is independent and writes only its own zarr subgroup,
            # so they parallelise cleanly across CPU-only processes (VS_DISABLE_GPU
            # keeps workers off DirectML -> no GPU contention). Workers are
            # idempotent (they skip a lattice that already has downsampled levels),
            # so a broken pool can safely fall back to a sequential pass.
            print(f"[segmentation] downsampling {len(jobs)} lattices "
                  f"with {workers} CPU worker process(es)")
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(_lattice_worker, job): job[1] for job in jobs}
                    for f in as_completed(futs):
                        f.result()  # re-raise any worker exception
            except Exception as e:  # pragma: no cover - resource dependent
                print(f"[segmentation] parallel run failed ({e!r}); "
                      f"finishing sequentially")
                for job in jobs:
                    _downsample_one_lattice(*job)
        else:
            for job in jobs:
                _downsample_one_lattice(*job)

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
    original_data: np.ndarray,
    downsampling_steps: int,
    ratios_to_be_stored: list[int],
    data_group: zarr.Group,
    value_to_seg_id: dict[int, int],
    params_for_storing: dict,
    time_frame: int,
):
    """
    GPU-accelerated equivalent of
    segmentation_downsampling._create_category_set_downsamplings.

    The per-level bookkeeping (a fresh ``current_set_table`` built from the
    ORIGINAL level-0 data on every step, popping level 0, filtering by ratio)
    is kept identical to the reference so the produced grids and set-tables
    match byte-for-byte. Only the inner per-block work is accelerated.
    """
    initial_set_table = SegmentationSetTable(original_data, value_to_seg_id)
    # The level-0 singleton table is invariant; compute it once and reuse it for
    # every level's fresh table instead of re-scanning the full grid each step.
    base_entries = initial_set_table.entries

    levels = [
        DownsamplingLevelDict(
            {"ratio": 1, "grid": original_data, "set_table": initial_set_table}
        )
    ]

    for i in range(downsampling_steps):
        current_set_table = SegmentationSetTable(
            None, value_to_seg_id, entries=base_entries
        )
        levels.append(
            downsample_categorical_data_gpu(levels[i], current_set_table)
        )

    levels.pop(0)
    levels = [lvl for lvl in levels if lvl.get_ratio() in ratios_to_be_stored]

    store_downsampling_levels_in_zarr(
        levels,
        lattice_data_group=data_group,
        params_for_storing=params_for_storing,
        time_frame=time_frame,
    )

    print(f"Stored GPU downsampling levels: {[lvl.get_ratio() for lvl in levels]}")


def _axis_indices(s: int):
    """
    Anchor/neighbour indices for every 2x2x2 output block along one axis.

    Target voxels are every other index 0, 2, 4, ... (matches
    MagicKernel3dDownsampler.extract_target_voxels_coords). The reference block is
    ``grid[start : min(start+2, s)]``, i.e. it includes the neighbour ``start+1``
    whenever it is in bounds and only falls back to the anchor for the last
    voxel of an odd-length axis. Duplicating the anchor there reproduces the
    reference's size-1 partial block exactly under a set union.
    """
    starts = np.arange(0, s, 2)
    idx0 = starts
    idx1 = np.minimum(starts + 1, s - 1)
    return idx0, idx1, int(starts.shape[0])


_GATHER_BUDGET_BYTES = 384 * 1024 * 1024  # peak GPU working set per output-X tile


def _gather_sort_torch(grid_np, xs, ys, zs):
    """
    Gather the 8 block corners of every output voxel and sort them, on GPU -
    memory-bounded so it works for arbitrarily large grids.

    We tile along the output-X axis and, for each tile, move only the thin slab
    of input-X planes that tile needs onto the GPU (never the whole grid, which
    would be ~2.5 GB as int64 for a 2048x2048 mask and OOM a memory-limited GPU).
    Both the slab and the (nvox_tile, 8) corner stack are kept within
    ``_GATHER_BUDGET_BYTES``. Tiles are contiguous output-X ranges processed in
    order and concatenated, so the row order is the exact C-order the reference
    iterates in - the result is bit-identical to a single-shot gather.

    The grid is cast to int64 per slab: DirectML fatally aborts on unsigned
    dtypes such as the uint32 the grids typically use, but int64 is safe and
    losslessly holds the small segment ids.
    """
    device = get_device()
    x0, x1 = xs
    y0, _y1 = ys
    z0, _z1 = zs
    sx = grid_np.shape[0]
    ny, nz = y0.shape[0], z0.shape[0]
    plane = grid_np.shape[1] * grid_np.shape[2]

    # full-resolution Y/Z index tensors (shared across tiles)
    ay_t = [torch.as_tensor(a.astype(np.int64)).to(device) for a in ys]
    az_t = [torch.as_tensor(a.astype(np.int64)).to(device) for a in zs]

    # per output-X: slab needs ~2 input planes; corner stack needs 8 int64 per
    # output voxel-column (ny*nz). Size the tile so both stay within budget.
    per_x = plane * 8 * 2 + ny * nz * 8 * 8
    tile = max(1, _GATHER_BUDGET_BYTES // max(1, per_x))

    nx = x0.shape[0]
    parts = []
    for start in range(0, nx, tile):
        end = min(start + tile, nx)
        # input-X slab covering this output-X tile (anchors 2*start.., neighbours +1)
        x_lo = int(x0[start])
        x_hi = min(int(2 * end), sx)
        slab = np.ascontiguousarray(grid_np[x_lo:x_hi]).astype(np.int64, copy=False)
        g = torch.as_tensor(slab).to(device)

        ax_local = [
            torch.as_tensor((ax[start:end] - x_lo).astype(np.int64)).to(device)
            for ax in (x0, x1)
        ]
        corners = []
        for ax in ax_local:
            for ay in ay_t:
                for az in az_t:
                    corners.append(
                        g[ax[:, None, None], ay[None, :, None], az[None, None, :]]
                    )
        stack = torch.stack(corners, dim=-1).reshape(-1, 8)
        parts.append(to_numpy(torch.sort(stack, dim=1).values))
        del g, stack, corners
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _gather_sort_numpy(grid_np, xs, ys, zs):
    """Corner gather + per-row sort on the CPU.

    This is the DEFAULT: the op is memory-bound, and on DirectML the GPU version
    (host round-trips + driver overhead) measured ~2-3x SLOWER than numpy here.
    """
    corners = []
    for ax in xs:
        for ay in ys:
            for az in zs:
                corners.append(grid_np[np.ix_(ax, ay, az)])
    stack = np.stack(corners, axis=-1).reshape(-1, 8)
    return np.sort(stack, axis=1)


def _unique_rows(rows: np.ndarray):
    """
    Unique over the rows of a 2D array, returning (representative_rows,
    first_occurrence_indices, inverse).

    Views each row as one opaque void scalar so ``np.unique`` runs a single 1-D
    sort instead of ``axis=0``'s multi-column lexsort. On ~100M rows this is an
    order of magnitude faster (measured ~44 s -> ~4 s) and bit-identical: the
    grouping of identical rows is the same, and the id-assignment below derives
    its order purely from ``first_idx``, independent of how uniques are sorted.
    """
    a = np.ascontiguousarray(rows)
    void = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _uq, first_idx, inverse = np.unique(void, return_index=True, return_inverse=True)
    return a[first_idx], first_idx, inverse.reshape(-1)


def _gather_and_unique_torch(grid_np, xs, ys, zs):
    """
    CUDA/ROCm path: gather the 8 block corners of every output voxel, sort each
    row, and dedup the rows - all on the GPU (device radix sort). Returns
    ``(unique_rows, first_occurrence_idx, inverse)`` on the host for the
    sequential id-assignment, matching numpy's
    ``np.unique(sig, return_index=True, return_inverse=True)`` semantics.

    ``first_idx`` is the first-occurrence index of each unique row (scatter-min
    of the position array over the inverse map) - the id-assignment only uses it
    to order groups by first appearance, so the arbitrary sort order of ``uniq``
    is irrelevant.

    NOTE: the *logic* is exercised on torch-CPU (see the correctness test); GPU
    throughput is unverified pending NVIDIA hardware. Guarded by a numpy fallback
    at the call site.
    """
    device = get_device()
    # keep the narrow (int8/int16) dtype from _narrow_labels - CUDA/ROCm/MPS all
    # handle signed ints, and smaller rows mean less memory + a faster sort.
    g = torch.as_tensor(np.ascontiguousarray(grid_np)).to(device)
    ax = [torch.as_tensor(a.astype(np.int64)).to(device) for a in xs]
    ay = [torch.as_tensor(a.astype(np.int64)).to(device) for a in ys]
    az = [torch.as_tensor(a.astype(np.int64)).to(device) for a in zs]

    corners = []
    for a in ax:            # order (x, y, z) == _gather_sort_numpy / reference
        for b in ay:
            for c in az:
                corners.append(g[a[:, None, None], b[None, :, None], c[None, None, :]])
    sig = torch.stack(corners, dim=-1).reshape(-1, 8)
    sig = torch.sort(sig, dim=1).values  # canonicalise each block's 8 values

    uniq, inverse = torch.unique(sig, dim=0, return_inverse=True)
    inverse = inverse.reshape(-1)

    n = sig.shape[0]
    first_idx = torch.full((uniq.shape[0],), n, dtype=torch.long, device=device)
    first_idx.scatter_reduce_(
        0, inverse, torch.arange(n, device=device), reduce="amin", include_self=True
    )
    return to_numpy(uniq), to_numpy(first_idx), to_numpy(inverse)


def _narrow_labels(grid: np.ndarray) -> np.ndarray:
    """
    Return ``grid`` viewed/cast to the smallest signed int dtype that still holds
    all its (non-negative) label values. Used only to shrink the gather stack -
    it never changes which values are equal, so signatures/grouping are identical.
    Non-integer or negative grids are returned unchanged.
    """
    if not np.issubdtype(grid.dtype, np.integer) or grid.size == 0:
        return grid
    mn, mx = int(grid.min()), int(grid.max())
    for dt in (np.int8, np.int16, np.int32):
        info = np.iinfo(dt)
        if info.min <= mn and mx <= info.max:
            return grid.astype(dt, copy=False)
    return grid


def downsample_categorical_data_gpu(
    previous_level_dict: DownsamplingLevelDict,
    current_set_table: SegmentationSetTable,
    prefer_gpu: bool = True,
) -> DownsamplingLevelDict:
    """
    Correct GPU/vectorized categorical-set downsampling.

    For every output voxel it forms the 2x2x2 block of the previous grid,
    takes the UNION of the segment-id sets those voxel-ids map to, and resolves
    that union to an id via ``current_set_table`` - exactly as the original
    ``downsample_categorical_data`` does. The expensive part (gathering the 8
    neighbours of each output voxel and canonicalising them) is vectorised on
    the GPU; the tiny, order-sensitive id-assignment stays on the CPU so the
    resulting ids are identical to the sequential reference.
    """
    previous_level_grid: np.ndarray = previous_level_dict.get_grid()
    previous_level_set_table: SegmentationSetTable = previous_level_dict.get_set_table()

    sx, sy, sz = previous_level_grid.shape
    x0, x1, nx = _axis_indices(sx)
    y0, y1, ny = _axis_indices(sy)
    z0, z1, nz = _axis_indices(sz)
    xs, ys, zs = (x0, x1), (y0, y1), (z0, z1)

    # Gather from the smallest int dtype that still holds every label. The block
    # signatures only need to compare/group equal values, so a narrower dtype is
    # exact - it just shrinks the (nvox, 8) stack (up to 4x less memory, the
    # enabler for running several masks in parallel) and speeds the void-view
    # sort (smaller keys). Output dtype is restored at the end.
    gather_grid = _narrow_labels(previous_level_grid)

    # On DirectML the corner gather/sort is memory-bound and measured slower than
    # numpy, so it defaults to CPU there. On CUDA/ROCm (mature GPU sort) we run the
    # whole gather + sort + row-dedup on the device (_gather_and_unique_torch);
    # only the tiny sequential id-assignment below stays on the host. Any device
    # failure (OOM, unsupported op) falls back to the numpy path.
    uniq = first_idx = inverse = None
    if prefer_gpu and TORCH_AVAILABLE and backend_name() in ("rocm/cuda", "mps"):
        try:
            uniq, first_idx, inverse = _gather_and_unique_torch(gather_grid, xs, ys, zs)
        except Exception as e:  # pragma: no cover - hardware/driver dependent
            print(f"[segmentation] GPU gather/unique failed ({e!r}); using CPU")
    if uniq is None:
        sig_host = _gather_sort_numpy(gather_grid, xs, ys, zs)
        uniq, first_idx, inverse = _unique_rows(sig_host)

    # Resolve ids by walking unique block-signatures in first-occurrence
    # (== block iteration) order, so brand-new categories get assigned the same
    # incrementing ids as the sequential reference.
    order = np.argsort(first_idx, kind="stable")
    prev_entries = previous_level_set_table.entries
    id_per_sig = np.empty(len(uniq), dtype=np.int64)
    union_to_id: dict = {}
    for si in order:
        distinct = set(int(v) for v in uniq[si])
        union = set().union(*(prev_entries[d] for d in distinct))
        key = frozenset(union)
        cid = union_to_id.get(key)
        if cid is None:
            cid = current_set_table.resolve_category(set(union))
            union_to_id[key] = cid
        id_per_sig[si] = cid

    current_level_grid = id_per_sig[inverse].reshape((nx, ny, nz)).astype(
        previous_level_grid.dtype
    )

    return DownsamplingLevelDict(
        {
            "ratio": round(previous_level_dict.get_ratio() * 2),
            "grid": current_level_grid,
            "set_table": current_set_table,
        }
    )

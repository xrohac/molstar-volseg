import numpy as np
import pyopencl as cl

from cellstar_preprocessor.flows.segmentation.downsampling_level_dict import DownsamplingLevelDict
from cellstar_preprocessor.flows.segmentation.segmentation_set_table import SegmentationSetTable
from cellstar_preprocessor.tools.magic_kernel_downsampling_3d.magic_kernel_downsampling_3d import MagicKernel3dDownsampler

# CPU help funkcia
def compute_union(block: np.ndarray, previous_table: SegmentationSetTable) -> set:
    block_values = tuple(block.flatten())
    categories = previous_table.get_categories(block_values)
    return set().union(*categories)

# OpenCL kernel na blokový downsampling
kernel_src = """
typedef unsigned long long u64;
u64 simple_hash(const int *vals) {
    u64 h = 146527;
    for (int i = 0; i < 8; i++)
        h ^= ((u64)vals[i] + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2));
    return h;
}
__kernel void downsample_block(
    __global const int *prev, __global int *curr,
    __global const u64 *keys, __global const int *vals,
    const int nk, const int sx, const int sy, const int sz) {
    int idx = get_global_id(0);
    int nx = sx/2, ny = sy/2, nz = sz/2;
    if (idx >= nx*ny*nz) return;
    int x = idx % nx, y = (idx / nx) % ny, z = idx / (nx*ny);
    int cx = x*2, cy = y*2, cz = z*2;
    int block[8];
    int sxy = sy*sx;
    int st = cz*sxy + cy*sx + cx;
    block[0] = prev[st+0];   block[1] = prev[st+1];
    block[2] = prev[st+sx];  block[3] = prev[st+sx+1];
    block[4] = prev[st+sxy]; block[5] = prev[st+sxy+1];
    block[6] = prev[st+sxy+sx]; block[7] = prev[st+sxy+sx+1];
    u64 h = simple_hash(block);
    int new_id = 0;
    for (int i = 0; i < nk; i++)
        if (keys[i] == h) { new_id = vals[i]; break; }
    curr[idx] = new_id;
}
"""

# Príprava hash mapy pre unikátne kombinácie (CPU)
def prepare_mapping(prev_grid: np.ndarray, prev_set_table: SegmentationSetTable):
    mapping: dict[int,int] = {}
    sx, sy, sz = prev_grid.shape
    for x in range(0, sx, 2):
        for y in range(0, sy, 2):
            for z in range(0, sz, 2):
                block = prev_grid[x:x+2, y:y+2, z:z+2]
                union = compute_union(block, prev_set_table)
                new_id = prev_set_table.resolve_category(union)
                key = hash(tuple(sorted(union)))
                mapping[key] = new_id
    keys = np.array(list(mapping.keys()), dtype=np.uint64)
    vals = np.array(list(mapping.values()), dtype=np.int32)
    return keys, vals

# Vlastná GPU downsample funkcia cez PyOpenCL
def gpu_downsample(prev_grid: np.ndarray, prev_set_table: SegmentationSetTable):
    sx, sy, sz = prev_grid.shape
    keys, vals = prepare_mapping(prev_grid.astype(int), prev_set_table)
    nx, ny, nz = sx//2, sy//2, sz//2

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kernel_src).build()

    mf = cl.mem_flags
    prev_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=prev_grid.astype(np.int32))
    curr_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=4 * nx * ny * nz)
    keys_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=keys)
    vals_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vals)

    total = nx * ny * nz
    prg.downsample_block(queue, (total,), None,
                         prev_buf, curr_buf, keys_buf, vals_buf,
                         np.int32(len(keys)),
                         np.int32(sx), np.int32(sy), np.int32(sz))

    out = np.empty(total, dtype=np.int32)
    cl.enqueue_copy(queue, out, curr_buf)
    return out.reshape((nx, ny, nz))

# Integrovaná GPU pipeline (nahrádza CPU downsample + ukladanie)
def process_downsampling_gpu(    
    magic_kernel: MagicKernel3dDownsampler,
    previous_level_dict: DownsamplingLevelDict,
    current_set_table: SegmentationSetTable,
    value_to_segment_id_dict_for_specific_lattice_id: dict
) -> DownsamplingLevelDict:

    prev_grid = previous_level_dict.get_grid()
    prev_table = previous_level_dict.get_set_table()

    new_grid = gpu_downsample(prev_grid, prev_table)
    new_set_table = SegmentationSetTable(new_grid.astype(int), value_to_segment_id_dict_for_specific_lattice_id)

    new_dict = DownsamplingLevelDict({
        "ratio": previous_level_dict.get_ratio() * 2,
        "grid": new_grid,
        "set_table": new_set_table
    })

    return new_dict
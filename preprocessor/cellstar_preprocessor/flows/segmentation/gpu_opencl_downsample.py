import numpy as np
import pyopencl as cl
from cellstar_preprocessor.flows.segmentation.segmentation_set_table import SegmentationSetTable

def prepare_mapping(previous_grid: np.ndarray, value_to_seg):
    table = SegmentationSetTable(previous_grid.astype(int), value_to_seg)
    mapping = {}
    # pre každú možnú kombináciu 8 hodnôt z blokov
    # tu jednoducho iterujeme cez všetky bloky (uprav podľa potreby)
    sx, sy, sz = previous_grid.shape
    for cx in range(0, sx, 2):
        for cy in range(0, sy, 2):
            for cz in range(0, sz, 2):
                block = previous_grid[cx:cx+2, cy:cy+2, cz:cz+2]
                vals = tuple(sorted(block.flatten()))
                new_id = table.resolve_category(set(vals))
                mapping[hash(vals)] = new_id
    keys = np.array(list(mapping.keys()), dtype=np.uint64)
    vals = np.array(list(mapping.values()), dtype=np.int32)
    return keys, vals

# OpenCL kernel
kernel_src = """
typedef unsigned long long u64;

u64 simple_hash(const int *vals) {
    u64 h = 146527;
    for (int i = 0; i < 8; i++)
        h ^= ((u64)vals[i]
               + 0x9e3779b97f4a7c15ULL
               + (h << 6) + (h >> 2));
    return h;
}

__kernel void downsample_block(
    __global const int *prev, __global int *curr,
    __global const u64 *keys, __global const int *vals,
    const int nk,
    const int sx, const int sy, const int sz)
{
    int idx = get_global_id(0);
    int nx = sx/2, ny = sy/2, nz = sz/2;
    if (idx >= nx * ny * nz) return;
    int x = idx % nx;
    int y = (idx / nx) % ny;
    int z = idx / (nx * ny);

    int cx = x*2, cy = y*2, cz = z*2;
    int block[8];
    int st = cz*sy*sx + cy*sx + cx;
    block[0] = prev[st];
    block[1] = prev[st+1];
    block[2] = prev[st+sx];
    block[3] = prev[st+sx+1];
    block[4] = prev[st+sy*sx];
    block[5] = prev[st+sy*sx+1];
    block[6] = prev[st+sy*sx+sx];
    block[7] = prev[st+sy*sx+sx+1];

    u64 h = simple_hash(block);
    int new_id = 0;
    for (int i = 0; i < nk; i++) {
        if (keys[i] == h) {
            new_id = vals[i];
            break;
        }
    }
    curr[idx] = new_id;
}
"""

def gpu_opencl_downsample(previous_grid: np.ndarray, value_to_seg: dict[int,int]):
    keys, vals = prepare_mapping(previous_grid, value_to_seg)
    sx, sy, sz = previous_grid.shape
    nx, ny, nz = sx//2, sy//2, sz//2

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    prg = cl.Program(ctx, kernel_src).build()

    mf = cl.mem_flags
    prev_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=previous_grid.astype(np.int32))
    curr_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=nx*ny*nz*4)
    keys_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=keys)
    vals_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vals)

    total = nx * ny * nz
    prg.downsample_block(queue, (total,), None,
                         prev_buf, curr_buf, keys_buf, vals_buf,
                         np.int32(len(keys)),
                         np.int32(sx), np.int32(sy), np.int32(sz))

    out_np = np.empty(total, dtype=np.int32)
    cl.enqueue_copy(queue, out_np, curr_buf)
    return out_np.reshape((nx, ny, nz))

# Použitie:
# previous_np = np.load(...)  # vstupná 3D matica
# result = gpu_opencl_downsample(previous_np, value_to_seg_id_dict)

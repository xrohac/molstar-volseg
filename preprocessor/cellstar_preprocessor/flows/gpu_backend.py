"""
Vendor-neutral GPU backend built on PyTorch.

The original preprocessing pipeline was written against CuPy / CUDA, which only
runs on NVIDIA hardware. This module replaces that hard CUDA dependency with a
PyTorch backend that automatically targets whatever accelerator is present, so
the exact same code runs on:

  * AMD GPUs on Windows   -> DirectML  (``pip install torch-directml``)
  * AMD GPUs on Linux     -> ROCm      (PyTorch reports this device as "cuda")
  * NVIDIA GPUs           -> CUDA
  * Apple Silicon         -> MPS
  * anything else         -> CPU       (always works, just slower)

None of the code that imports this module is CUDA-specific: to move between
vendors you only change which PyTorch build is installed, not the source.

Every helper degrades gracefully - if PyTorch (or a working GPU) is missing,
callers can check ``TORCH_AVAILABLE`` / ``gpu_is_available()`` and fall back to
the pure-CPU reference implementations.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - depends on runtime install
    torch = None
    TORCH_AVAILABLE = False


_DEVICE = None
_BACKEND_NAME = "numpy-cpu"
_CONV2D_OK: bool | None = None


def _detect_device():
    """Pick the fastest available PyTorch device, preferring real GPUs."""
    global _BACKEND_NAME

    # Explicit opt-out - used by CPU worker processes so they never initialize a
    # GPU context (avoids DirectML contention when several run in parallel).
    if os.environ.get("VS_DISABLE_GPU"):
        _BACKEND_NAME = "cpu" if TORCH_AVAILABLE else "numpy-cpu"
        return torch.device("cpu") if TORCH_AVAILABLE else None

    if not TORCH_AVAILABLE:
        _BACKEND_NAME = "numpy-cpu"
        return None

    # NVIDIA CUDA *and* AMD ROCm both surface here - on a ROCm build of PyTorch
    # ``torch.cuda`` transparently drives AMD cards.
    try:
        if torch.cuda.is_available():
            _BACKEND_NAME = "rocm/cuda"
            return torch.device("cuda")
    except Exception:
        pass

    # AMD (and Intel) GPUs on Windows go through DirectML.
    try:
        import torch_directml

        if torch_directml.is_available():
            _BACKEND_NAME = "directml"
            return torch_directml.device()
    except Exception:
        pass

    # Apple Silicon.
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            _BACKEND_NAME = "mps"
            return torch.device("mps")
    except Exception:
        pass

    _BACKEND_NAME = "cpu"
    return torch.device("cpu")


def get_device():
    """Return the cached PyTorch device (or ``None`` when torch is unavailable)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = _detect_device()
    return _DEVICE


def backend_name() -> str:
    """Human-readable name of the selected backend, e.g. ``directml``."""
    get_device()
    return _BACKEND_NAME


def gpu_is_available() -> bool:
    """True when a non-CPU PyTorch device was selected."""
    if not TORCH_AVAILABLE:
        return False
    return backend_name() not in ("cpu", "numpy-cpu")


def conv2d_supported() -> bool:
    """
    Probe whether the selected device can actually run ``conv2d``.

    We build 3D volume downsampling out of ``conv2d`` (see
    ``volume_downsampling_gpu``) rather than ``conv3d``, because some backends -
    notably DirectML on Windows - implement 2D convolution but not 3D. We run a
    tiny strided conv2d once and cache the result so callers can cleanly fall
    back to the CPU pipeline when even 2D convolution is unavailable.
    """
    global _CONV2D_OK
    if _CONV2D_OK is not None:
        return _CONV2D_OK
    if not TORCH_AVAILABLE:
        _CONV2D_OK = False
        return _CONV2D_OK

    try:
        device = get_device()
        x = torch.zeros((1, 1, 4, 4), dtype=torch.float32, device=device)
        w = torch.ones((1, 1, 2, 2), dtype=torch.float32, device=device)
        y = torch.nn.functional.conv2d(x, w, stride=2)
        _ = float(y.sum().detach().to("cpu"))
        _CONV2D_OK = True
    except Exception:
        _CONV2D_OK = False
    return _CONV2D_OK


def to_device(array, dtype=None):
    """Move a numpy array onto the selected device as a torch tensor."""
    tensor = torch.as_tensor(np.ascontiguousarray(array))
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.to(get_device())


def to_device_safe(array):
    """
    Move a numpy array to the device using only widely-supported dtypes.

    Some backends (notably DirectML) *fatally abort the process* - not a catchable
    Python exception - when handed an unsupported dtype such as ``uint32``. We
    therefore normalize on the host first: floating-point -> float32, everything
    else (integer/bool, including unsigned) -> int64. Both are safe on every
    backend, and int64 losslessly holds the small non-negative segment ids used
    by the categorical paths.
    """
    a = np.ascontiguousarray(array)
    if np.issubdtype(a.dtype, np.floating):
        return torch.as_tensor(a.astype(np.float32, copy=False)).to(get_device())
    return torch.as_tensor(a.astype(np.int64, copy=False)).to(get_device())


def to_numpy(tensor) -> np.ndarray:
    """Bring a torch tensor back to host memory as a numpy array."""
    return tensor.detach().to("cpu").numpy()

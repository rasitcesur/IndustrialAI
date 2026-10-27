"""Optional CUDA multi-stream implementation of the *legacy Cython* feature operator.

This is deliberately NOT the earlier OpenCV-CUDA fused-kernel implementation:
that algorithm changed the second branch (kernel3) and border semantics.
CUDA kernels here implement the precise graph used by fast_kernels.pyx.

OpenCV's separately compiled CUDA module is not required; a CUDA-capable
PyTorch wheel, compatible NVIDIA driver and GPU are required.
"""
from __future__ import annotations

import numpy as np


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except (ImportError, OSError, RuntimeError):
        return False


def _torch_raw_maps(x, gaussian, kernel0, kernel1, neighbor):
    """Return pre-activation (minimum, maximum) maps, torch on CPU or CUDA.

    x: float32 NCHW grayscale tensor; H/W must be divisible by 16.
    Mirrors legacy fast_kernels.pyx, including clamping pixel coordinates at
    borders for K0 and K3 and using UNSMOOTHED conv10 neighbor values for K3.
    """
    import torch
    import torch.nn.functional as F

    height, width = x.shape[-2:]
    if height < 16 or width < 16 or height % 16 or width % 16:
        raise ValueError('Image dimensions must be multiples of 16 and >=16')
    # Cython pass order: horizontal, then vertical. REFLECT_101 on both axes.
    x = F.conv2d(F.pad(x, (7, 7, 0, 0), mode='reflect'), gaussian['horizontal'])
    x = F.conv2d(F.pad(x, (0, 0, 7, 7), mode='reflect'), gaussian['vertical'])
    # Cython explicitly sets conv10 border to zero.
    interior = F.conv2d(x, kernel1)
    conv10 = F.pad(interior, (1, 1, 1, 1), mode='constant', value=0)
    # K0 and neighbor sum are evaluated at clamped coordinates (1..H-2).
    k0 = F.conv2d(conv10, kernel0, padding=1)
    neighbors = F.conv2d(conv10, neighbor, padding=1)
    ys = torch.arange(height, device=x.device).clamp(1, height - 2)
    xs = torch.arange(width, device=x.device).clamp(1, width - 2)
    k0 = k0.index_select(-2, ys).index_select(-1, xs)
    neighbors = neighbors.index_select(-2, ys).index_select(-1, xs)
    k3 = 5.0 * k0 - neighbors
    mins = -F.max_pool2d(-k0, kernel_size=16, stride=16)
    maxs = F.max_pool2d(k3, kernel_size=16, stride=16)
    return mins, maxs


class AsyncCudaFeatureExtractor:
    """Overlapped pinned H2D -> CUDA convolution/pooling -> pinned D2H.

    CUDA concurrency may help large batches, but is NOT assumed faster for small
    512x512 images. Each stream has independent input/output buffers. The CPU
    performs the final two activations to match the reference output closely.
    """
    def __init__(self, img_h=512, img_w=512, num_streams=4):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA-enabled PyTorch and NVIDIA GPU are required')
        if (img_h, img_w) != (512, 512):
            raise ValueError('Research model requires 512x512 images')
        if not (1 <= num_streams <= 16):
            raise ValueError('num_streams must be between 1 and 16')
        # Disable TF32 convolution to preserve numerical equivalence with CPU float32.
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.height, self.width = img_h, img_w
        self.num_streams = num_streams
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(num_streams)]
        self.in_cpu = [torch.empty((img_h, img_w), dtype=torch.uint8, pin_memory=True)
                       for _ in range(num_streams)]
        self.out_cpu = [torch.empty((2048,), dtype=torch.float32, pin_memory=True)
                        for _ in range(num_streams)]
        self.in_gpu = [torch.empty((1, 1, img_h, img_w), dtype=torch.uint8, device=self.device)
                       for _ in range(num_streams)]
        self.f32_gpu = [torch.empty((1, 1, img_h, img_w), dtype=torch.float32, device=self.device)
                        for _ in range(num_streams)]
        from cv2 import getGaussianKernel
        g = getGaussianKernel(15, 0).astype(np.float32).ravel()
        # float32 Gaussian, zero border of K1 and original neighbor K3 semantics
        self.gaussian = {
            'horizontal': torch.as_tensor(g.reshape(1, 1, 1, 15).copy(), device=self.device),
            'vertical': torch.as_tensor(g.reshape(1, 1, 15, 1).copy(), device=self.device),
        }
        self.kernel0 = torch.as_tensor(np.asarray([
            [1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]],
            dtype=np.float32).reshape(1, 1, 3, 3).copy(), device=self.device)
        self.kernel1 = torch.as_tensor(np.asarray([
            [0, -1, 0], [-1, 3, -1], [0, -1, 0]],
            dtype=np.float32).reshape(1, 1, 3, 3).copy(), device=self.device)
        self.neighbor = torch.as_tensor(np.asarray([
            [0, 1, 0], [1, 0, 1], [0, 1, 0]],
            dtype=np.float32).reshape(1, 1, 3, 3).copy(), device=self.device)

    def _enqueue(self, img, slot):
        t = self.torch
        if img.shape != (self.height, self.width) or img.dtype != np.uint8:
            raise ValueError('Expected 512x512 grayscale uint8 image')
        # Critical: do not reuse pinned slots until the matching CUDA stream has completed.
        np.copyto(self.in_cpu[slot].numpy(), img)
        with t.cuda.stream(self.streams[slot]):
            self.in_gpu[slot].copy_(self.in_cpu[slot].view(1, 1, self.height, self.width),
                                    non_blocking=True)
            self.f32_gpu[slot].copy_(self.in_gpu[slot], non_blocking=True)
            mins, maxs = _torch_raw_maps(self.f32_gpu[slot], self.gaussian,
                                          self.kernel0, self.kernel1, self.neighbor)
            pooled_raw = t.cat((mins.reshape(-1), maxs.reshape(-1)))
            self.out_cpu[slot].copy_(pooled_raw, non_blocking=True)

    def _collect(self, slot, output, index):
        self.streams[slot].synchronize()
        raw = self.out_cpu[slot].numpy()
        lo = (raw[:1024] - np.float32(127.5)) * np.float32(1 / 127.5)
        hi = raw[1024:] * np.float32(1 / 255)
        # Stable transformations, float32 output; never use test labels.
        output[index, :1024] = 1.0 / (1.0 + np.exp(-lo))
        output[index, 1024:] = np.logaddexp(np.float32(0), hi)

    def process_batch(self, images):
        if isinstance(images, np.ndarray):
            arr = images
        else:
            arr = np.asarray(images)
        if arr.ndim != 3 or arr.shape[1:] != (self.height, self.width) or arr.dtype != np.uint8:
            raise ValueError('Images must have shape (N,512,512) and uint8 dtype')
        out = np.empty((len(arr), 2048), dtype=np.float32)
        for i, img in enumerate(arr):
            slot = i % self.num_streams
            if i >= self.num_streams:
                self._collect(slot, out, i - self.num_streams)
            self._enqueue(img, slot)
        for i in range(max(0, len(arr) - self.num_streams), len(arr)):
            self._collect(i % self.num_streams, out, i)
        return out


def cuda_self_test(tolerance=3e-4):
    """Compare against the authoritative Python legacy operator before GPU use.

    Raise instead of using a CUDA backend that changes the trained features.
    """
    from .features import base_features_python
    rng = np.random.default_rng(721)
    a = rng.integers(0, 256, (512,512), dtype=np.uint8)
    b = np.zeros((512,512), dtype=np.uint8)
    b[:,256:] = 255
    c = np.tile(np.arange(512, dtype=np.uint16).clip(max=255).astype(np.uint8), (512,1))
    stack = np.stack([a,b,c])
    extractor = AsyncCudaFeatureExtractor(num_streams=2)
    got = extractor.process_batch(stack)
    expected = np.stack([base_features_python(image) for image in stack])
    max_abs = float(np.max(np.abs(got - expected)))
    if not np.isfinite(got).all() or max_abs > tolerance:
        raise RuntimeError(f'CUDA parity test failed: max abs error {max_abs:.8g}, limit {tolerance}')
    return max_abs

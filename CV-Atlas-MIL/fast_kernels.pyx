# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

from libc.math cimport exp, log
from libc.stdlib cimport malloc, free
from cython.parallel import prange

# Extracts pooled convolutional features using separable 1D Gaussian filters
# and fused L1/L2-cache-aligned sliding windows across OpenMP threads.
#
# @param images 3D contiguous uint8 array of shape (N, H, W).
# @param g_kernel 1D contiguous float32 array for 15x15 Gaussian filter.
# @param out_features 2D contiguous float32 pre-allocated output array.
cpdef void extract_features_batch(
    const unsigned char[:, :, ::1] images,
    const float[::1] g_kernel,
    float[:, ::1] out_features
) noexcept nogil:
    cdef Py_ssize_t n_imgs = images.shape[0]
    cdef Py_ssize_t H = images.shape[1]
    cdef Py_ssize_t W = images.shape[2]
    cdef Py_ssize_t blocks_h = H // 16
    cdef Py_ssize_t blocks_w = W // 16
    cdef Py_ssize_t half_blocks = blocks_h * blocks_w
    cdef Py_ssize_t i

    for i in prange(n_imgs, schedule='static'):
        _process_single_image(
            images[i], g_kernel, out_features[i], H, W, blocks_h, blocks_w, half_blocks
        )


# Internal worker executing separable convolution and fused pooling for a single image.
# Allocates thread-local heap buffers to prevent OpenMP stack overflow on Windows.
#
# @param img 2D uint8 view of source image.
# @param g_kernel 1D float32 Gaussian weights (size 15).
# @param out_row 1D float32 view of target feature vector.
# @param H Image height.
# @param W Image width.
# @param blocks_h Number of vertical 16x16 pooling blocks.
# @param blocks_w Number of horizontal 16x16 pooling blocks.
# @param half_blocks Offset for concatenating minPool and maxPool vectors.
cdef inline void _process_single_image(
    const unsigned char[:, ::1] img,
    const float[::1] g_kernel,
    float[::1] out_row,
    Py_ssize_t H,
    Py_ssize_t W,
    Py_ssize_t blocks_h,
    Py_ssize_t blocks_w,
    Py_ssize_t half_blocks
) noexcept nogil:
    cdef Py_ssize_t y, x, k, by, bx, py, px, ny, nx, feat_idx
    cdef Py_ssize_t row_offset, n_row_offset
    cdef float acc, k0_val, k3_val, min_v, max_v, norm_min, norm_max

    # Allocate flat 1D contiguous buffers per thread (fits inside L2/L3 cache)
    cdef float* temp_h = <float*>malloc(H * W * sizeof(float))
    cdef float* blurred = <float*>malloc(H * W * sizeof(float))
    cdef float* conv10 = <float*>malloc(H * W * sizeof(float))

    if temp_h == NULL or blurred == NULL or conv10 == NULL:
        if temp_h != NULL:
            free(temp_h)
        if blurred != NULL:
            free(blurred)
        if conv10 != NULL:
            free(conv10)
        return

    # Step 1: Horizontal 1D Gaussian Pass (Cache-friendly row-major traversal)
    for y in range(H):
        row_offset = y * W
        for x in range(W):
            acc = 0.0
            for k in range(-7, 8):
                nx = x + k
                if nx < 0:
                    nx = -nx
                elif nx >= W:
                    nx = 2 * W - nx - 2
                acc = acc + (<float>img[y, nx]) * g_kernel[k + 7]
            temp_h[row_offset + x] = acc

    # Step 2: Vertical 1D Gaussian Pass
    for y in range(H):
        row_offset = y * W
        for x in range(W):
            acc = 0.0
            for k in range(-7, 8):
                ny = y + k
                if ny < 0:
                    ny = -ny
                elif ny >= H:
                    ny = 2 * H - ny - 2
                acc = acc + temp_h[ny * W + x] * g_kernel[k + 7]
            blurred[row_offset + x] = acc

    # Initialize borders to zero
    for y in range(H):
        conv10[y * W] = 0.0
        conv10[y * W + (W - 1)] = 0.0
    for x in range(W):
        conv10[x] = 0.0
        conv10[(H - 1) * W + x] = 0.0

    # Step 3: Kernel 1 Pass [0, -1, 0; -1, 3, -1; 0, -1, 0]
    for y in range(1, H - 1):
        row_offset = y * W
        for x in range(1, W - 1):
            acc = (
                3.0 * blurred[row_offset + x]
                - blurred[(y - 1) * W + x]
                - blurred[(y + 1) * W + x]
                - blurred[row_offset + (x - 1)]
                - blurred[row_offset + (x + 1)]
            )
            conv10[row_offset + x] = acc

    # Step 4: Block-Tiled Pooling (16x16) + Kernel 0 + Kernel 3 + Activations
    feat_idx = 0
    for by in range(blocks_h):
        for bx in range(blocks_w):
            min_v = 1e9
            max_v = -1e9
            for py in range(by * 16, (by + 1) * 16):
                for px in range(bx * 16, (bx + 1) * 16):
                    ny = py if py > 0 else 1
                    ny = ny if ny < H - 1 else H - 2
                    nx = px if px > 0 else 1
                    nx = nx if nx < W - 1 else W - 2

                    n_row_offset = ny * W

                    # Kernel 0 smoothing response
                    k0_val = (
                        0.25 * conv10[n_row_offset + nx]
                        + 0.125 * (
                            conv10[(ny - 1) * W + nx]
                            + conv10[(ny + 1) * W + nx]
                            + conv10[n_row_offset + (nx - 1)]
                            + conv10[n_row_offset + (nx + 1)]
                        )
                        + 0.0625 * (
                            conv10[(ny - 1) * W + (nx - 1)]
                            + conv10[(ny - 1) * W + (nx + 1)]
                            + conv10[(ny + 1) * W + (nx - 1)]
                            + conv10[(ny + 1) * W + (nx + 1)]
                        )
                    )

                    if k0_val < min_v:
                        min_v = k0_val

                    # Kernel 3 sharpening response
                    k3_val = 5.0 * k0_val - (
                        conv10[(ny - 1) * W + nx]
                        + conv10[(ny + 1) * W + nx]
                        + conv10[n_row_offset + (nx - 1)]
                        + conv10[n_row_offset + (nx + 1)]
                    )
                    if k3_val > max_v:
                        max_v = k3_val

            norm_min = (min_v - 127.5) * (1.0 / 127.5)
            norm_max = max_v * (1.0 / 255.0)

            out_row[feat_idx] = exp(norm_min) / (1.0 + exp(norm_min))
            out_row[feat_idx + half_blocks] = log(1.0 + exp(norm_max))
            feat_idx = feat_idx + 1

    free(temp_h)
    free(blurred)
    free(conv10)
#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* results,
                         unsigned int width,
                         unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters
) {
    // Ctrl + C, Ctrl + V    :()
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    unsigned int idx_x = get_global_id(0);
    unsigned int idx_y = get_global_id(1);

    if (idx_x < width && idx_y < height) {
        float x0 = fromX + (idx_x + 0.5f) * sizeX / width;
        float y0 = fromY + (idx_y + 0.5f) * sizeY / height;

        float x = x0;
        float y = y0;

        int iter = 0;
        for (; iter < iters; ++iter) {
            float xPrev = x;
            x = x * x - y * y + x0;
            y = 2.0f * xPrev * y + y0;
            if ((x * x + y * y) > threshold2) {
                break;
            }
        }

        results[idx_y * width + idx_x] = 1.0f * iter / iters;
    }
}

#define TILE_SIZE 32
__kernel void matrix_multiplication(
        __global const float* as_gpu,
        __global const float* bs_gpu,
        __global float* cs_gpu,
        const unsigned M,
        const unsigned K,
        const unsigned N)
{
    unsigned loc_idx_x = get_local_id(0);
    unsigned loc_idx_y = get_local_id(1);
    unsigned glb_idx_x = get_global_id(0);
    unsigned glb_idx_y = get_global_id(1);

    // Будем пытаться переиспользовать подгруженные в кэш части
    // Заведем 2 таблицы (по текущему рассмт-у куску для каждого из операндов)
    __local float A[TILE_SIZE * TILE_SIZE];
    __local float B[TILE_SIZE * TILE_SIZE];

    for(unsigned i = 0; i < K + TILE_SIZE; i += TILE_SIZE) {  // i -- отступ текущих тайлов
        // Подгрузим тайлы в локальную память
        unsigned A_idx_x = i + loc_idx_x;
        unsigned A_idx_y = glb_idx_y;
        unsigned B_idx_x = glb_idx_x;
        unsigned B_idx_y = i + loc_idx_y;

        if (A_idx_y < M && A_idx_x < K) {
            A[loc_idx_y * TILE_SIZE + loc_idx_x] = as_gpu[A_idx_y * K + A_idx_x];
        }
        if (B_idx_y < K && B_idx_x < N) {
            B[loc_idx_y * TILE_SIZE + loc_idx_x] = bs_gpu[B_idx_y * N + B_idx_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // Ждем запись в локальную память

        if (glb_idx_x < N && glb_idx_y < M) {
            for (unsigned j = 0; j < TILE_SIZE; ++j) {
                // Считаем ячейку (glb_idx_x, glb_idx_y)
                cs_gpu[glb_idx_y * N + glb_idx_x] += A[loc_idx_y * TILE_SIZE + j] * B[j * TILE_SIZE + loc_idx_x];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // Ждем расчет текущих тайлов, чтобы не было неожиданной перезаписи лок. памяти
    }
}
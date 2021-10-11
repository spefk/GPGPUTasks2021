#define TILE_SIZE 32
__kernel void matrix_transpose(
        __global float* as_gpu,
        __global float* as_t_gpu,
        unsigned int rows,
        unsigned int cols
) {
    unsigned int gl_idx_x = get_global_id(0);
    unsigned int gl_idx_y = get_global_id(1);
    unsigned int loc_idx_x = get_local_id(0);
    unsigned int loc_idx_y = get_local_id(1);

    // Таблица для промежуточной записи "квадрата" значений
    __local float buffer_table [TILE_SIZE * (TILE_SIZE + 1)];  // Не инициализируем, т.к. нет смысла

    // Заполняем таблицу
    if (gl_idx_x < cols && gl_idx_y < rows) {
        buffer_table[loc_idx_y * (TILE_SIZE + 1) + loc_idx_x] = as_gpu[gl_idx_x + gl_idx_y * cols];
    }

    unsigned int corner_x = TILE_SIZE * get_group_id(0);  // Посчитаем углы, чтобы проще было записывать по строкам
    unsigned int corner_y = TILE_SIZE * get_group_id(1);  // (т.е. чтобы было coalesced)

    barrier(CLK_LOCAL_MEM_FENCE);
    // Читаем из таблицы
    if (gl_idx_x < cols && gl_idx_y < rows) {
        as_t_gpu[(corner_y + loc_idx_x) + (corner_x + loc_idx_y) * rows] =
                buffer_table[loc_idx_x * (TILE_SIZE + 1) + loc_idx_y];
    }

    // TODO: Можно использовать транспонирование для coalesced доступа
    // TODO: Можно в локальной памяти решать bank-конфликты (с пом сдвигов или поворотов строк)

}
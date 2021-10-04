#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256
__kernel void sum(__global unsigned int* numbers,
                  __global unsigned int* res,
                  unsigned int n
) {
    unsigned int local_idx = get_local_id(0);
    unsigned int group_idx = get_group_id(0);

    unsigned int fst = group_idx * WORK_GROUP_SIZE; // Индекс первого числа для этой группы
    __local int mask[WORK_GROUP_SIZE];  // Маска, был ли уже сложен i-ый элемент
    if (local_idx % 2 == 0) {
        mask[fst + local_idx] = 1;          // инициализируем 1-ми
    } else {
        mask[fst + local_idx] = 0;          // инициализируем 0-ми
    }
    barrier(CLK_LOCAL_MEM_FENCE);       // Ждем пока все инициализируется

    // Пусть каждый воркер в группе складывает 2 элемента
    for (int i = 1; i < WORK_GROUP_SIZE - 1; i *= 2) {  // i -- окно, разность позиций суммируемых элементов
        unsigned int lhs = fst + local_idx;
        unsigned int rhs = fst + local_idx + i;
        if (mask[rhs - fst] != 1 && lhs < n && rhs < n) {  // Проверяем, был ли уже сложен элемент rhs
            // TODO: кажется есть проблема со входом нечетной длинны
            // TODO: да и в целом проблема с индексами :C
            numbers[lhs] += numbers[rhs];
            mask[rhs - fst] = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // Ждем пока все сложат, иначе получим ub в сумме
    };

    atomic_add(res, numbers[fst]);
}

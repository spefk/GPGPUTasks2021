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
    unsigned int fst = get_group_id(0) * WORK_GROUP_SIZE; // Индекс первого числа для этой группы

    // Пусть каждый воркер в группе складывает 2 элемента
    // Будем "складывать" текущий отрезок "пополам"
    for (int i = WORK_GROUP_SIZE / 2; i >= 1; i /= 2) {
        if (
                local_idx < i &&                     // Попали в нужный кусок
                local_idx + i + fst < n              // Не вылезли ли за пределы массива
        ) { numbers[fst + local_idx] += numbers[fst + local_idx + i]; }
        barrier(CLK_LOCAL_MEM_FENCE);                // Ждем пока все сложат, иначе получим ub в сумме
    };

    if (local_idx == 0) { atomic_add(res, numbers[fst]); }
}

// TODO: кажется, есть проблема со входом нечетной длинны, и ее лучше ловить вне GPU :)


#include "tile_program.hpp"

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"

#include "ck/library/utility/device_memory.hpp"

// program
struct GemmMultiplD
{
    __host__ __device__ void operator()(TileProgram& tp, int x, int y)
    {
        auto desc = tp.make_naive_tensor_descriptor_packed(ck::make_tuple(x));

        printf("length %d\n", desc.GetLength(ck::Number<0>{}));
    }
};

int main()
{
    int x = 100;
    int y = 101;

    launch(HelloWorld{}, 1, 1, x, y);

    return 0;
}

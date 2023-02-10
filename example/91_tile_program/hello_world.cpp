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
struct HelloWorld
{
    __host__ __device__ void operator()(TileProgram& tp, int x, int y, int* res)
    {
        auto desc0 = tp(make_naive_tensor_descriptor_packed(ck::make_tuple(x)));
        auto desc1 = tp(make_naive_tensor_descriptor_packed(ck::make_tuple(y)));

        res[0] = desc0.GetLength(ck::Number<0>{});
        res[1] = desc1.GetLength(ck::Number<0>{});
    }
};

int main()
{
    int x = 100;
    int y = 101;

    DeviceMem res_dev_buf(2 * sizeof(int));

    launch(HelloWorld{}, 1, 1, x, y, static_cast<int*>(res_dev_buf.GetDeviceBuffer()));

    int res_host[2];

    res_dev_buf.FromDevice(res_host);

    printf("res_host %d\n", res_host[0]);
    printf("res_host %d\n", res_host[1]);

    return 0;
}

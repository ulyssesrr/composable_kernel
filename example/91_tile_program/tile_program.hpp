
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

// namespace tp (for tile programming)
struct TileProgram
{
    // hidden intermediate argument
    struct Arg
    {
        char data_[1024];
        ck::index_t size_ = 0;
    };

    // arg on device
    Arg arg_;
    ck::index_t arg_pos_ = 0;

    // push arg on host
    template <typename T>
    __host__ auto push_arg(const T& a)
    {
        *reinterpret_cast<T*>(arg_.data_ + arg_.size_) = a;

        arg_.size_ += sizeof(T);

        return a;
    }

    // pull arg on device
    template <typename T>
    __device__ T pull_arg()
    {
        auto a = *reinterpret_cast<T*>(arg_.data_ + arg_pos_);

        arg_pos_ += sizeof(T);

        return a;
    }

    // host push
    template <typename... Lengths>
    __host__ constexpr auto
    make_naive_tensor_descriptor_packed(const ck::Tuple<Lengths...>& lengths)
    {
        auto desc = ck::make_naive_tensor_descriptor_packed(lengths);

        return push_arg(desc);
    }

    // device pull
    template <typename... Lengths>
    __device__ constexpr auto
    make_naive_tensor_descriptor_packed(const ck::Tuple<Lengths...>& lengths)
    {
        using Desc = decltype(ck::make_naive_tensor_descriptor_packed(lengths));

        return pull_arg<Desc>();
    }
};

template <typename Program, typename... Xs>
__global__ void gpu_program_wrapper(Program f, TileProgram tp, Xs... xs)
{
    f(tp, xs...);
}

template <typename Program, typename... Xs>
void launch(Program f, dim3 grid_dim, dim3 block_dim, Xs... xs)
{
    TileProgram tp;

    f(tp, xs...);

    printf("cpu arg size %d\n", tp.arg_.size_);

    gpu_program_wrapper<Program><<<grid_dim, block_dim, 0, nullptr>>>(f, tp, xs...);
}

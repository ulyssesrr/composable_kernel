
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

// hidden intermediate argument
struct Arg
{
    char data_[128];
    ck::index_t size_ = 0;
    ck::index_t pos_  = 0;

    __host__ __device__ void reset()
    {
        size_ = 0;
        pos_  = 0;
    }

    __device__ void reset_pos() { pos_ = 0; }

    // push arg on host
    template <typename T>
    __host__ T push(const T& a)
    {
        *reinterpret_cast<T*>(data_ + size_) = a;

        size_ += sizeof(T);

        return a;
    }

    // pull arg on device
    template <typename T>
    __device__ T pull()
    {
        T a = *reinterpret_cast<T*>(data_ + pos_);

        pos_ += sizeof(T);

        return a;
    }
};

// namespace tp (for tile programming)
struct TileProgram
{
    // arg on device
    Arg arg_;

    __device__ void gpu_init() { arg_.reset_pos(); }

    // push arg on host
    template <typename T>
    __host__ T operator()(const T& a)
    {
        return arg_.push(a);
    }

    // push arg on host
    template <typename T>
    __device__ T operator()(const T&)
    {
        return arg_.pull<T>();
    }

    __host__ static ck::index_t get_block_1d_id() { return -1; }

    __host__ static ck::index_t get_grid_size() { return -1; }

    __device__ static ck::index_t get_block_1d_id() { return ck::get_block_1d_id(); }

    __device__ static ck::index_t get_grid_size() { return ck::get_grid_size(); }
};

template <typename Program, typename... Xs>
__global__ void gpu_program_wrapper(Program f, TileProgram tp, Xs... xs)
{
    tp.gpu_init();
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

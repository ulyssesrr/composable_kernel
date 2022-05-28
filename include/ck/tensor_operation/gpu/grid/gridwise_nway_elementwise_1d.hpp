#pragma once

#include "cluster_descriptor.hpp"
#include "data_type.hpp"
#include "element_wise_operation.hpp"
#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename GridwiseEltwise,
          typename SrcDataTypes,
          typename DstDataTypes,
          typename SrcGridDesc_M,
          typename DstGridDesc_M,
          typename ElementwiseFunctor>
__global__ void kernel_nway_elementwise_1d(const SrcDataTypes p_src_globals,
                                           DstDataTypes p_dst_globals,
                                           const SrcGridDesc_M src_grid_desc_ms,
                                           const DstGridDesc_M dst_grid_desc_ms,
                                           const ElementwiseFunctor functor)
{
    GridwiseEltwise::Run(p_src_globals, p_dst_globals, src_grid_desc_ms, dst_grid_desc_ms, functor);
}

template <typename SrcDataTypes,
          typename DstDataTypes,
          typename ComputeDataType,
          typename SrcGridDesc_M,
          typename DstGridDesc_M,
          typename ElementwiseFunctor,
          index_t MPerThread,
          typename SrcScalarPerVector,
          typename DstScalarPerVector>
struct GridwiseNWayElementwise_1D
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto thread_desc_m =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MPerThread>{}));

    using PassThrough = tensor_operation::element_wise::PassThrough;

    static __device__ auto CalculateElementwiseIndex()
    {
        const index_t global_thread_id = get_thread_global_1d_id();
        return make_multi_index(global_thread_id * MPerThread);
    }

    __device__ static void Run(const SrcDataTypes p_src_globals,
                               DstDataTypes p_dst_globals,
                               const SrcGridDesc_M src_grid_desc_ms,
                               const DstGridDesc_M dst_grid_desc_ms,
                               const ElementwiseFunctor functor)
    {
        constexpr auto Isrc_size = Number<SrcDataTypes::Size()>{};
        constexpr auto Idst_size = Number<DstDataTypes::Size()>{};

        const auto src_global_buf = generate_tuple(
            [&](auto I) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_src_globals[I], src_grid_desc_ms[I].GetElementSpaceSize());
            },
            Isrc_size);

        auto dst_global_buf = generate_tuple(
            [&](auto I) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_dst_globals[I], dst_grid_desc_ms[I].GetElementSpaceSize());
            },
            Idst_size);

        auto src_thread_buf = generate_tuple(
            [&](auto) {
                return StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MPerThread, true>{};
            },
            Isrc_size);

        auto dst_thread_buf = generate_tuple(
            [&](auto) {
                return StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MPerThread, true>{};
            },
            Idst_size);
        const auto thread_store_global_offset = CalculateElementwiseIndex();

        auto src_global_load = generate_tuple(
            [&](auto I) {
                auto p_src_global      = p_src_globals[I];
                auto p_src_grid_desc_m = src_grid_desc_ms[I];

                return ThreadwiseTensorSliceTransfer_v2<
                    remove_const_t<remove_pointer_t<decltype(p_src_global)>>,
                    ComputeDataType,
                    decltype(p_src_grid_desc_m),
                    decltype(thread_desc_m),
                    Sequence<MPerThread>,      // SliceLengths
                    Sequence<0>,               // DimAccessOrder
                    0,                         // SrcVectorDim
                    SrcScalarPerVector::At(I), // ScalarPerVector
                    1,                         // SrcScalarStrideInVector
                    false>{p_src_grid_desc_m, thread_store_global_offset};
            },
            Isrc_size);

        auto dst_global_write = generate_tuple(
            [&](auto I) {
                auto p_dst_global      = p_dst_globals[I];
                auto p_dst_grid_desc_m = dst_grid_desc_ms[I];

                return ThreadwiseTensorSliceTransfer_v1r3<
                    ComputeDataType,
                    remove_pointer_t<decltype(p_dst_global)>,
                    decltype(thread_desc_m),
                    decltype(p_dst_grid_desc_m),
                    PassThrough,
                    Sequence<MPerThread>,      // SliceLengths
                    Sequence<0>,               // DimAccessOrder
                    0,                         // DstVectorDim
                    DstScalarPerVector::At(I), // ScalarPerVector
                    InMemoryDataOperationEnum::Set,
                    1, // DstScalarStrideInVector
                    false>{p_dst_grid_desc_m, thread_store_global_offset, PassThrough{}};
            },
            Idst_size);

        const index_t blockSize    = get_block_size();
        const index_t blockPerGrid = get_grid_size();
        const auto M               = dst_grid_desc_ms[I0].GetLength(I0);
        const index_t loop_step    = blockPerGrid * blockSize * MPerThread;
        const auto loop_step_index = make_multi_index(loop_step);

        index_t num_iter = M / (loop_step);
        do
        {
            // read and process MPerThread elements
            static_for<0, Isrc_size, 1>{}([&](auto I) {
                src_global_load(I).Run(src_grid_desc_ms[I],
                                       src_global_buf[I],
                                       thread_desc_m,
                                       make_tuple(I0),
                                       src_thread_buf(I));

                src_global_load(I).MoveSrcSliceWindow(src_grid_desc_ms[I], loop_step_index);
            });

            static_for<0, MPerThread, 1>{}([&](auto m) {
                constexpr auto offset = thread_desc_m.CalculateOffset(make_tuple(m));
                const auto src_tuple  = generate_tuple(
                    [&](auto I) { return src_thread_buf[I][Number<offset>{}]; }, Isrc_size);

                auto dst_tuple = generate_tuple(
                    [&](auto I) { return dst_thread_buf(I)(Number<offset>{}); }, Idst_size);

                (void)src_tuple;
                (void)dst_tuple;
                // TODO - n-ary functor
                // functor(src_tuple, dst_tuple);
            });

            static_for<0, Idst_size, 1>{}([&](auto I) {
                dst_global_write(I).Run(thread_desc_m,
                                        make_tuple(I0), // SrcSliceOriginIdx
                                        dst_thread_buf[I],
                                        dst_grid_desc_ms[I],
                                        dst_global_buf(I));

                dst_global_write(I).MoveDstSliceWindow(dst_grid_desc_ms[I], loop_step_index);
            });
        } while(--num_iter);
    }
};

} // namespace ck

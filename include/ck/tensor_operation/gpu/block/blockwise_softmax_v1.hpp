// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"

namespace ck {
template <index_t BlockSize,
          typename AccDataType,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t RegSizePerXdlops,
          index_t MRepeat,
          index_t NRepeat,
          index_t MThreadSliceSize,
          index_t NThreadSliceSize>
struct BlockwiseSoftmax_V1
{
    static constexpr auto I0            = Number<0>{};
    static constexpr auto I1            = Number<1>{};
    static constexpr auto I2            = Number<2>{};
    constexpr static auto c_thread_desc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, Number<RegSizePerXdlops>{}));
    template <typename CThreadBuffer>
    __host__ __device__ static void Run(CThreadBuffer& c_thread_buf)
    {
        // printf("c_thread_desc: {%d, %d, %d}", c_thread_desc.GetLength(I0).value,
        //    c_thread_desc.GetLength(I1).value, c_thread_desc.GetLength(I2).value);
        __shared__ AccDataType p_reduce_work_buffer[BlockSize];

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> max_value_buf;
        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            max_value_buf(I) = reduce::Max::template GetIdentityValue<AccDataType>();
        });

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> accu_value_buf;
        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accu_value_buf(I) = reduce::Add::template GetIdentityValue<AccDataType>();
        });

        constexpr index_t c_offset = c_thread_desc.CalculateOffset(make_tuple(0, 0, 0));
        auto& xdlops_out           = c_thread_buf.GetVectorTypeReference(Number<c_offset>{});

        using ThreadReduceSrcDesc_M_K = decltype(make_naive_tensor_descriptor_packed(
            make_tuple(Number<1>{}, Number<c_thread_desc.GetLength(I2)>{})));
        using ThreadReduceDstDesc_M =
            decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<1>{})));

        using ThreadwiseMaxReduce =
            ThreadwiseReduction<AccDataType,
                                ThreadReduceSrcDesc_M_K,
                                ThreadReduceDstDesc_M,
                                reduce::Max,
                                false, // param ignored
                                detail::AccumulateWithNanIgnore<reduce::Max, AccDataType>>;
        ThreadwiseMaxReduce::Reduce(xdlops_out.template AsType<float>(), max_value_buf);
        // const index_t thread_local_id = get_thread_local_1d_id();
        // printf("thread id: %d, Max: %f\t\t",thread_local_id,max_value_buf[I0]);

        using ThreadClusterLengths_M_K  = Sequence<32, 2>;
        using ThreadClusterArrangeOrder = Sequence<1, 0>;
        using BlockwiseMaxReduce        = PartitionedBlockwiseReduction<
            AccDataType,
            BlockSize,
            ThreadClusterLengths_M_K,
            ThreadClusterArrangeOrder,
            reduce::Max,
            false, // param ignored
            detail::AccumulateWithNanIgnore<reduce::Max, AccDataType>>;

        auto reduce_work_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_buffer, BlockSize);
        block_sync_lds();
        BlockwiseMaxReduce::Reduce(reduce_work_buf, max_value_buf(I0));
        block_sync_lds();

        // printf("\n");
        // printf("thread id: %d, Max: %f\t\t",thread_local_id,max_value_buf[I0]);
        // softmax
        using BlockwiseSumReduce = PartitionedBlockwiseReduction<
            AccDataType,
            BlockSize,
            ThreadClusterLengths_M_K,
            ThreadClusterArrangeOrder,
            reduce::Add,
            false, // ignored
            detail::AccumulateWithNanIgnore<reduce::Add, AccDataType>>;

        using ThreadwiseSumReduce =
            ThreadwiseReduction<AccDataType,
                                ThreadReduceSrcDesc_M_K,
                                ThreadReduceDstDesc_M,
                                reduce::Add,
                                false, // ignored
                                detail::AccumulateWithNanIgnore<reduce::Add, AccDataType>>;
        static_for<0, c_thread_desc.GetLength(I2), 1>{}([&](auto iK) {
            xdlops_out.template AsType<float>()(iK) =
                math::exp(xdlops_out.template AsType<float>()[iK] - max_value_buf(I0));
        });
        ThreadwiseSumReduce::Reduce(xdlops_out.template AsType<float>(), accu_value_buf);
        block_sync_lds();
        BlockwiseSumReduce::Reduce(reduce_work_buf, accu_value_buf(I0));
        block_sync_lds();
        static_for<0, c_thread_desc.GetLength(I2), 1>{}([&](auto iK) {
            xdlops_out.template AsType<float>()(iK) =
                xdlops_out.template AsType<float>()[iK] / accu_value_buf(I0);
        });
    }
};

} // namespace ck

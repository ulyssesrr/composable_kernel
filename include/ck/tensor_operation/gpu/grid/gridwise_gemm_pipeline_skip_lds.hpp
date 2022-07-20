// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {

__device__ void s_nop()
{
#if 1
    asm volatile("\
    s_nop 0 \n \
    " ::);
#else
    __builtin_amdgcn_sched_barrier(0);
#endif
}

struct GridwiseGemmPipelineSkipBLds
{
    __host__ __device__ static constexpr bool IsSupported(index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop >= 2;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 2;
    }

    template <bool HasMainLoop,
              index_t MultK0,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BThreadDesc,
              typename BThreadTransfer,
              typename BGridBuffer,
              typename BThreadBuffer,
              typename BThreadTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BThreadDesc& b_thread_desc,
                               BThreadTransfer& b_threadwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BThreadBuffer* b_thread_buf,
                               const BThreadTransferStep& b_thread_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        // preload data to regiester and LDS
        // Read
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

        // Move
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_slice_copy_step);


        static_for<0, MultK0, 1>{}([&](auto i_load_b){
            b_threadwise_copy.Run(b_grid_desc,
                                  b_grid_buf,
                                  b_thread_desc,
                                  make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                  b_thread_buf[i_load_b]);

            s_nop();
            b_threadwise_copy.MoveSrcSliceWindow(b_grid_desc,
                                                 b_thread_slice_copy_step);
        });

        // Initialize C
        c_thread_buf.Clear();
        // a data write to lds
        a_blockwise_copy.RunWrite(a_block_desc_k0_m_k1, a_block_buf);

        // main body
        if constexpr(HasMainK0BlockLoop)
        {
            index_t i = 0;
            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                blockwise_gemm.ResetABlockStartWindow();
                block_sync_lds();

                static_for<0, MultiK0, 1>{}([&](auto i_main) {

                    blockwise_gemm.Run(a_block_buf, b_thread_buf[i_main], c_thread_buf);
                    // 1st
                    b_threadwise_copy.Run(b_grid_desc,
                                          b_grid_buf,
                                          b_thread_desc,
                                          make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                          b_thread_buf[i_main]);
                    b_threadwise_copy.MoveSrcSliceWindow(b_grid_desc,
                                                         b_thread_slice_copy_step);
                    blockwise_gemm.MoveABlockSliceWindow();
                    s_nop();
                });

                block_sync_lds();
                a_blockwise_copy.RunWrite(a_block_desc_k0_m_k1, a_block_buf);
                // move a and b window
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc,
                                                    a_block_slice_copy_step);

                i += 1;
            } while(i < (num_loop - 1));

            // tail
            {
                block_sync_lds();

                blockwise_gemm.ResetABlockStartWindow();
                static_for<0, MultiK0, 1>{}([&](auto i_tail) {
                    

                    blockwise_gemm.Run(a_block_buf, b_thread_buf[i_tail], c_thread_buf);
                    blockwise_gemm.MoveABlockSliceWindow();


                    blockwise_gemm.Run(a_block_buf, b_thread_buf[i_tail], c_thread_buf);
                    blockwise_gemm.MoveABlockSliceWindow();
                });
            }
        }
    }
};

struct GridwiseGemmPipelineAInVgpr
{
    static constexpr I0 = Number<0>{};
    __host__ __device__ static constexpr bool IsSupported(index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop >= 2;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 2;
    }

    template<typename TypeConvertFp32ToFp16Functor, typename AThreadDesc, index_t i_k0>
    __device__ static void ConvertCopy(const AThreadDesc& a_thread_desc, 
                                       const AccThreadBuffer& acc_thread_buf, 
                                       AThreadBuffer& a_thread_buf)
    {
        constexpr auto i_k0_num = Number<i_k0>{};
        static_for<0, n0, 1>{}([&](auto n) {
            static_for<0, m4, 1>{}([&](auto m) {
                constexpr auto acc_offset = a_thread_desc.CalculateOffset(make_tuple(i_k0_num, n, I0, I0, I0, I0, m, I0));
                constexpr auto a_offset = a_thread_desc.CalculateOffset(make_tuple(I0, n, I0, I0, I0, I0, m, I0));
                TypeConvertFp32ToFp16Functor(a1_thread_buf(Number<a_offset>{}), acc_thread_buf(Number<acc_offset>{}));
            });
        });
    }

    template <typename TypeConvertFp32ToFp16Functor,
              index_t MultK0,
              typename AThreadDesc,
              typename AccThreadBuffer,
              typename AThreadBuffer,
              typename AThreadTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AThreadDesc& a_thread_desc,
                               const AccThreadBuffer& acc_thread_buf,
                               AThreadBuffer& a_thread_buf,
                               const AThreadTransferStep& a_thread_transfer_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf)
    {
        static_for<0, MultiK0, 1>{}([&](auto i_k0){
            
        });

        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();
        // a data write to lds
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        // main body
        if constexpr(HasMainK0BlockLoop)
        {
            index_t i = 0;
            do
            {
                block_sync_lds();

                // GEMM i
                blockwise_gemm.Run(a_thread_buf, b_block_buf, c_thread_buf);

                block_sync_lds();
                
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                // LDS write i + 1
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
                // global read i + 2
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                i += 1;
            } while(i < (num_loop - 2));

            // tail
            {
                block_sync_lds();

                // GEMM num_loop - 2
                blockwise_gemm.Run(a_thread_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                // LDS write num_loop - 1
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                block_sync_lds();

                // GEMM num_loop - 1
                blockwise_gemm.Run(a_thread_buf, b_block_buf, c_thread_buf);

            }
        }
    }
};

} // namespace ck

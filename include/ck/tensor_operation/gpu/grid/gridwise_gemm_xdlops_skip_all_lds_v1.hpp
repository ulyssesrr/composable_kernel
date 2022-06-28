#pragma once

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "blockwise_gemm_xdlops.hpp"
#include "blockwise_gemm_xdlops_skip_b_lds.hpp"
#include "blockwise_gemm_xdlops_skip_all_lds.hpp"
#include "thread_group_tensor_slice_transfer_v4r1.hpp"
#include "thread_group_tensor_slice_transfer_v6r1.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "gridwise_gemm_pipeline_v1.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_K0_M_K1,
          typename AGridDesc_K0_K1_K2_M0_M1_M2_M3_K3,
          typename BGridDesc_K0_N_K1,
          typename BGridDesc_K0_K1_K2_N0_N1_N2_N3_K3,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainK0BlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_xdlops_skip_all_lds_v1(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1,
            const AGridDesc_K0_K1_K2_M0_M1_M2_M3_K3 a_grid_desc_k0_k1_k2_m0_m1_m2_m3_k3,
            const BGridDesc_K0_K1_K2_N0_N1_N2_N3_K3 b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3,
            const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                                      c_grid_desc_mblock_mperblock_nblock_nperblock,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op,
            const Block2CTileMap block_2_ctile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<HasMainK0BlockLoop>(
        p_a_grid,
        p_b_grid,
        p_c_grid,
        p_shared,
        a_grid_desc_k0_m_k1,
        a_grid_desc_k0_k1_k2_m0_m1_m2_m3_k3,
        b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3,
        c_grid_desc_mblock_mperblock_nblock_nperblock,
        a_element_op,
        b_element_op,
        c_element_op,
        block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = block_2_ctile_map;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename CGridDesc_M_N,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MultiK0,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t K1Value,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          bool ABlockLdsExtraM,
          index_t BBlockTransferSrcScalarPerVector,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          index_t CBlockTransferScalarPerVector_NWaveNPerXDL,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>
struct GridwiseGemm_k0mk1_k0nk1_mn_xdlops_skip_all_lds_v1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    //static constexpr auto MultiK0 = 16 * 1;

    // K1 should be Number<...>
    static constexpr auto K1 = Number<K1Value>{};

    static constexpr index_t WaveSize = 64;
    static constexpr index_t MWaves   = MPerBlock / (MXdlPerWave * MPerXDL);
    static constexpr index_t NWaves   = NPerBlock / (NXdlPerWave * NPerXDL);

    static constexpr auto xdlops_gemm    = XdlopsGemm<FloatAB, MPerXDL, NPerXDL, K1>{};
    static constexpr index_t K0PerThread = K0PerBlock / xdlops_gemm.K0PerXdlops;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto c_block_size =
            GetCBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock().GetElementSpaceSize();

        return (c_block_size) * sizeof(FloatC);
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
                  const BGridDesc_K0_N_K1& b_grid_desc_k0_n_k1,
                  const CGridDesc_M_N& c_grid_desc_m_n,
                  const Block2CTileMap& block_2_ctile_map)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerXDL * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXDL)) == 0,
                      "Invalid tuning param!");

        const auto M  = a_grid_desc_k0_m_k1.GetLength(I2);
        const auto N  = b_grid_desc_k0_n_k1.GetLength(I2);
        const auto K0 = a_grid_desc_k0_m_k1.GetLength(I1);

        if(!(M == c_grid_desc_m_n.GetLength(I0) && N == c_grid_desc_m_n.GetLength(I1) &&
             K0 == b_grid_desc_k0_n_k1.GetLength(I1) && K1 == a_grid_desc_k0_m_k1.GetLength(I3) &&
             K1 == b_grid_desc_k0_n_k1.GetLength(I3)))
            return false;

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K0 % K0PerBlock == 0))
            return false;

        // 2-stage prefetch currently only support even number of K0 loop
        // TODO: add support for odd number of K0 loop
        if(!((K0 / K0PerBlock) % 2 == 0))
        {
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(c_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr index_t
    CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const index_t grid_size = (M / MPerBlock) * (N / NPerBlock);

        return grid_size;
    }

    // TODO move this function into GEMM-pipeline class
    __host__ __device__ static constexpr bool CalculateHasMainK0BlockLoop(index_t K0)
    {
        const bool has_main_k0_block_loop = K0 > (MultiK0 * K0PerBlock);

        return has_main_k0_block_loop;
    }

    // TODO move this function into GEMM-pipeline class
    __host__ __device__ static constexpr index_t CalculateResMainK0BlockLoop(index_t K0)
    {
        const index_t res_main_k0_block_loop = (K0 / K0PerBlock) % MultiK0;

        return res_main_k0_block_loop;
    }

    __host__ __device__ static constexpr auto
    MakeAGridDescriptor_K0_K1_K2_M0_M1_M2_M3_K3(const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1)
    {
        const auto KBatch = a_grid_desc_k0_m_k1.GetLength(I0);
        const auto K0 = a_grid_desc_k0_m_k1.GetLength(I1);
        const auto M  = a_grid_desc_k0_m_k1.GetLength(I2);

        const auto a_griddesc_k0_mblockid_mrepeat_mwaves_mperxdlops_k1 =
            transform_tensor_descriptor(
                a_grid_desc_k0_m_k1,
                make_tuple(make_pass_through_transform(KBatch),
                           make_unmerge_transform(
                               make_tuple(K0 / K0PerBlock, xdlops_gemm.K0PerXdlops, K0PerThread)),
                           make_unmerge_transform(make_tuple(
                               M / (MXdlPerWave * MWaves * MPerXDL), MXdlPerWave, MWaves, MPerXDL)),
                           make_pass_through_transform(K1)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4, 5, 6, 7>{}, Sequence<8>{}));
        return a_griddesc_k0_mblockid_mrepeat_mwaves_mperxdlops_k1;
    }

    __host__ __device__ static constexpr auto
    MakeBGridDescriptor_K0_K1_K2_N0_N1_N2_N3_K3(const BGridDesc_K0_N_K1& b_grid_desc_k0_n_k1)
    {
        const auto KBatch = b_grid_desc_k0_n_k1.GetLength(I0);
        const auto K0 = b_grid_desc_k0_n_k1.GetLength(I1);
        const auto N  = b_grid_desc_k0_n_k1.GetLength(I2);

        const auto b_griddesc_k0_nblockid_nrepeat_nwaves_nperxdlops_k1 =
            transform_tensor_descriptor(
                b_grid_desc_k0_n_k1,
                make_tuple(make_pass_through_transform(KBatch),
                           make_unmerge_transform(
                               make_tuple(K0 / K0PerBlock, xdlops_gemm.K0PerXdlops, K0PerThread)),
                           make_unmerge_transform(make_tuple(
                               N / (NXdlPerWave * NWaves * NPerXDL), NXdlPerWave, NWaves, NPerXDL)),
                           make_pass_through_transform(K1)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4, 5, 6, 7>{}, Sequence<8>{}));
        return b_griddesc_k0_nblockid_nrepeat_nwaves_nperxdlops_k1;
    }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = get_thread_local_1d_id();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto GetWaveKNIdx(const index_t thread_id)
    {
        constexpr auto wave_threadid_to_nk_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(xdlops_gemm.K0PerXdlops, NPerXDL))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return wave_threadid_to_nk_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto GetWaveKMIdx(const index_t thread_id)
    {
        constexpr auto wave_threadid_to_mk_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(xdlops_gemm.K0PerXdlops, MPerXDL))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return wave_threadid_to_mk_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __host__ __device__ static constexpr auto
    MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(const CGridDesc_M_N& c_m_n_grid_desc)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        return transform_tensor_descriptor(
            c_m_n_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto MakeCBlockClusterAdaptor(
        const CGridDesc_M_N & c_m_n_grid_desc, index_t /* M01 */, index_t /* N01 */, index_t KBatch)
    {
        return BlockToCTileMap_KSplit_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
            c_m_n_grid_desc, 8, KBatch);
    }

    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXDL);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXDL);

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1,
                       Number<CShuffleMRepeatPerShuffle * MWave * MPerXDL>{},
                       I1,
                       Number<CShuffleNRepeatPerShuffle * NWave * NPerXDL>{}));
    }

    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(CGridDesc_M_N{}));
    using CBlockClusterAdaptor = decltype(MakeCBlockClusterAdaptor(CGridDesc_M_N{}, 1, 1, 1));

    using BGridDesc_K0_K1_K2_N0_N1_N2_N3_K3 =
        decltype(MakeBGridDescriptor_K0_K1_K2_N0_N1_N2_N3_K3(BGridDesc_K0_N_K1{}));

    using AGridDesc_K0_K1_K2_M0_M1_M2_M3_K3 =
        decltype(MakeAGridDescriptor_K0_K1_K2_M0_M1_M2_M3_K3(AGridDesc_K0_M_K1{}));

    template <bool HasMainK0BlockLoop>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        void* __restrict__ p_shared,
        const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
        const AGridDesc_K0_K1_K2_M0_M1_M2_M3_K3 a_grid_desc_k0_k1_k2_m0_m1_m2_m3_k3,
        const BGridDesc_K0_K1_K2_N0_N1_N2_N3_K3 b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3,
        const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CElementwiseOperation& c_element_op,
        const CBlockClusterAdaptor& c_block_cluster_adaptor)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_k0_k1_k2_m0_m1_m2_m3_k3.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        const auto K0 = a_grid_desc_k0_m_k1.GetLength(I1);

        //const auto ResMainK0BlockLoop = CalculateResMainK0BlockLoop(K0);

        // divide block work by [M, N]
        const auto block_work_idx =
            c_block_cluster_adaptor.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        //const index_t k_batch_id = block_work_idx[I0];


        // A matrix blockwise copy
        // a thread wise copy
        ignore = a_element_op;
        constexpr auto a_thread_desc_k0_k1_k2_m0_m1_m2_m3_k3 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           I1,
                                                           I1,
                                                           Number<K0PerThread>{}, // K0PerThread
                                                           I1,                    // NBlockId
                                                           Number<MXdlPerWave>{}, // repeat
                                                           I1,                    // waves
                                                           I1,                    // NPerXdlops
                                                           Number<K1>{}));

        auto a_thread_buf = generate_tuple(
            [&](auto i) {
                ignore = i;
                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    FloatAB,
                                    a_thread_desc_k0_k1_k2_m0_m1_m2_m3_k3.GetElementSpaceSize(),
                                    true>{};
            },
            Number<MultiK0>{});
        //StaticBuffer<AddressSpaceEnum::Vgpr,
        //             FloatAB,
        //             a_thread_desc_k0_k1_k2_m0_m1_m2_m3_k3.GetElementSpaceSize(),
        //             true> 
        //    a_thread_buf[MultiK0];

        ignore = b_element_op;
        // B matrix threadwise copy
        constexpr auto b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           I1,
                                                           I1,
                                                           Number<K0PerThread>{}, // K0PerThread
                                                           I1,                    // NBlockId
                                                           Number<NXdlPerWave>{}, // repeat
                                                           I1,                    // waves
                                                           I1,                    // NPerXdlops
                                                           Number<K1>{}));

        auto b_thread_buf = generate_tuple(
            [&](auto i) {
                ignore = i;
                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    FloatAB,
                                    b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetElementSpaceSize(),
                                    true>{};
            },
            Number<MultiK0>{});

        //StaticBuffer<AddressSpaceEnum::Vgpr,
        //             FloatAB,
        //             b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetElementSpaceSize(),
        //             true> 
        //    b_thread_buf[MultiK0];

        const auto wave_id     = GetWaveIdx();
        const auto wave_k_n_id = GetWaveKNIdx(wave_id[I2]);

        const auto wave_k_m_id = GetWaveKMIdx(wave_id[I2]);

#if 0
        const index_t block_id  = get_block_1d_id();
        const index_t thread_id = get_thread_local_1d_id();
        printf("block id: %d kbatch id: %d m blockid: %d n block id: %d ,thread id: %d, wave id :{%d %d %d} "
               "kn id: {%d %d}, km id: {%d %d}\n",
               block_id,
               block_work_idx[I0],
               block_work_idx[I1],
               block_work_idx[I2],
               thread_id,
               wave_id[I0],
               wave_id[I1],
               wave_id[I2],
               wave_k_n_id[I0],
               wave_k_n_id[I1],
               wave_k_m_id[I0],
               wave_k_m_id[I1]);
        printf("mfma thread k per xdlops: %d K0PerThread: %d HasMainK0BlockLoop: %d KBatch: %d K0: %d \n", 
                xdlops_gemm.K0PerXdlops, K0PerThread, HasMainK0BlockLoop, 
                b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetLength(I0), 
                b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3.GetLength(I1));
#endif
        auto a_threadwise_copy =
            ThreadwiseTensorSliceTransfer_v2<FloatAB,
                                             FloatAB,
                                             decltype(a_grid_desc_k0_k1_k2_m0_m1_m2_m3_k3),
                                             decltype(a_thread_desc_k0_k1_k2_m0_m1_m2_m3_k3),
                                             Sequence<I1,
                                                      I1,
                                                      I1,
                                                      Number<K0PerThread>{},
                                                      I1,
                                                      Number<MXdlPerWave>{},
                                                      I1,
                                                      I1,
                                                      Number<K1>{}>,
                                             Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>,
                                             8,
                                             ABlockTransferSrcScalarPerVector,
                                             AThreadTransferSrcResetCoordinateAfterRun,
                                             true>(
                a_grid_desc_k0_k1_k2_m0_m1_m2_m3_k3,
                make_multi_index(
                    block_work_idx[I0], 0, wave_k_m_id[I0], 0, block_work_idx[I1], 0, wave_id[I1], wave_k_m_id[I1], 0));

        auto b_threadwise_copy =
            ThreadwiseTensorSliceTransfer_v2<FloatAB,
                                             FloatAB,
                                             decltype(b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3),
                                             decltype(b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3),
                                             Sequence<I1,
                                                      I1,
                                                      I1,
                                                      Number<K0PerThread>{},
                                                      I1,
                                                      Number<NXdlPerWave>{},
                                                      I1,
                                                      I1,
                                                      Number<K1>{}>,
                                             Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>,
                                             8,
                                             BBlockTransferSrcScalarPerVector,
                                             BThreadTransferSrcResetCoordinateAfterRun,
                                             true>(
                b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3,
                make_multi_index(
                    block_work_idx[I0], 0, wave_k_n_id[I0], 0, block_work_idx[I2], 0, wave_id[I1], wave_k_n_id[I1], 0));

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check
        auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_non_lds<
            BlockSize,
            FloatAB,
            FloatAcc,
            decltype(a_thread_desc_k0_k1_k2_m0_m1_m2_m3_k3),
            decltype(b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3),
            MPerBlock,
            NPerBlock,
            K0PerBlock,
            MPerXDL,
            NPerXDL,
            MXdlPerWave,
            NXdlPerWave,
            K1>{};

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // gridwise GEMM pipeline
        // constexpr auto a_block_slice_copy_step  = make_multi_index(K0PerBlock * MultiK0, 0, 0);
        constexpr auto a_thread_slice_copy_step = make_multi_index(0, 1, 0, 0, 0, 0, 0, 0, 0);
        constexpr auto b_thread_slice_copy_step = make_multi_index(0, 1, 0, 0, 0, 0, 0, 0, 0);

        if constexpr(HasMainK0BlockLoop)
        {
            // Read
            static_for<0, MultiK0, 1>{}([&](auto i_pre) {
                b_threadwise_copy.Run(b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3,
                                      b_grid_buf,
                                      b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3,
                                      make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                      b_thread_buf(Number<i_pre>{}));
                a_threadwise_copy.Run(a_grid_desc_k0_k1_k2_m0_m1_m2_m3_k3,
                                      a_grid_buf,
                                      a_thread_desc_k0_k1_k2_m0_m1_m2_m3_k3,
                                      make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                      a_thread_buf(Number<i_pre>{}));

                asm volatile("s_nop 0" ::);

                // Move
                b_threadwise_copy.MoveSrcSliceWindow(b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3,
                                                     b_thread_slice_copy_step);
                a_threadwise_copy.MoveSrcSliceWindow(a_grid_desc_k0_k1_k2_m0_m1_m2_m3_k3,
                                                     a_thread_slice_copy_step);
            });

            // Initialize C
            c_thread_buf.Clear();

            // main body
            {
                index_t K0BlockMainLoop = __builtin_amdgcn_readfirstlane(K0 / K0PerBlock);
                index_t i               = 0;
                do
                {
                    static_for<0, MultiK0, 1>{}([&](auto i_k) {
                        blockwise_gemm.Run(a_thread_buf(Number<i_k>{}), b_thread_buf(Number<i_k>{}), c_thread_buf);

                        asm volatile("s_nop 0" ::);

                        b_threadwise_copy.Run(b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3,
                                              b_grid_buf,
                                              b_thread_desc_k0_k1_k2_n0_n1_n2_n3_k3,
                                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                              b_thread_buf(Number<i_k>{}));
                        a_threadwise_copy.Run(a_grid_desc_k0_k1_k2_m0_m1_m2_m3_k3,
                                              a_grid_buf,
                                              a_thread_desc_k0_k1_k2_m0_m1_m2_m3_k3,
                                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                              a_thread_buf(Number<i_k>{}));

                        asm volatile("s_nop 0" ::);

                        b_threadwise_copy.MoveSrcSliceWindow(b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3,
                                                             b_thread_slice_copy_step);
                        a_threadwise_copy.MoveSrcSliceWindow(a_grid_desc_k0_k1_k2_m0_m1_m2_m3_k3,
                                                             a_thread_slice_copy_step);
                    });

                    i += MultiK0;
                } while(i < (K0BlockMainLoop - MultiK0));
            }

            // tail
            {
                //index_t loop_num = ResMainK0BlockLoop == 0 ? MultiK0 : ResMainK0BlockLoop;
                static_for<0, MultiK0, 1>{}([&](auto i) {
                    blockwise_gemm.Run(
                        a_thread_buf(Number<i>{}), b_thread_buf(Number<i>{}), c_thread_buf);
                });
            }
        }

#if 1
        // output: register to global memory
        {
            constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXDL);
            constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXDL);

            constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc =
                blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc =
                blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto M0 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I0);
            constexpr auto N0 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I1);
            constexpr auto M1 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I2);
            constexpr auto N1 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I3);
            constexpr auto M2 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I4);
            constexpr auto M3 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I5);
            constexpr auto M4 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I6);
            constexpr auto N2 = c_m0_n0_m1_n1_m2_m3_m4_n2_block_desc.GetLength(I7);

            constexpr auto c_block_desc_mblock_mperblock_nblock_nperblock =
                GetCBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<FloatC*>(p_shared),
                c_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            static_assert(M1 == MWave, "");
            static_assert(N1 == NWave, "");
            static_assert(M2 * M3 * M4 == MPerXDL, "");
            static_assert(N2 == NPerXDL, "");

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = transform_tensor_descriptor(
                c_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0), // freeze mblock
                    make_unmerge_transform(make_tuple(CShuffleMRepeatPerShuffle,
                                                      M1,
                                                      M2,
                                                      M3,
                                                      M4)), // M1 = MWave, M2 * M3 * M4 = MPerXDL
                    make_freeze_transform(I0),              // freeze nblock
                    make_unmerge_transform(make_tuple(CShuffleNRepeatPerShuffle,
                                                      N1,
                                                      N2))), // M1 = MWave, M2 * M3 * M4 = MPerXDL
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4, 5, 6>{}, Sequence<>{}, Sequence<1, 3, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            // VGPR to LDS
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatAcc,
                                                   FloatC,
                                                   decltype(c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   Sequence<CShuffleMRepeatPerShuffle,
                                                            CShuffleNRepeatPerShuffle,
                                                            I1,
                                                            I1,
                                                            M2,
                                                            I1,
                                                            M4,
                                                            I1>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     m_thread_data_on_block_idx[I3],
                                     m_thread_data_on_block_idx[I4],
                                     n_thread_data_on_block_idx[I2]),
                    ck::tensor_operation::element_wise::PassThrough{}};

            // LDS to global
            auto c_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // index_t BlockSize,
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMRepeatPerShuffle * MWave * MPerXDL,
                         1,
                         CShuffleNRepeatPerShuffle * NWave * NPerXDL>, // BlockSliceLengths,
                CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                FloatC,               // typename SrcData,
                FloatC,               // typename DstData,
                decltype(c_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                Sequence<0, 1, 2, 3>,                       // typename DimAccessOrder,
                3,                                          // index_t VectorDim,
                CBlockTransferScalarPerVector_NWaveNPerXDL, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun
                {c_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 c_grid_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(block_work_idx[I1], 0, block_work_idx[I2], 0),
                 c_element_op};

            constexpr auto mxdlperwave_forward_step =
                make_multi_index(0, CShuffleMRepeatPerShuffle * MWave * MPerXDL, 0, 0);
            constexpr auto nxdlperwave_forward_step =
                make_multi_index(0, 0, 0, CShuffleNRepeatPerShuffle * NWave * NPerXDL);
            constexpr auto nxdlperwave_backward_step =
                make_multi_index(0, 0, 0, -CShuffleNRepeatPerShuffle * NWave * NPerXDL);

            static_for<0, MXdlPerWave, CShuffleMRepeatPerShuffle>{}([&](auto mxdlperwave_iter) {
                constexpr auto mxdlperwave = mxdlperwave_iter;

                static_for<0, NXdlPerWave, CShuffleNRepeatPerShuffle>{}([&](auto nxdlperwave_iter) {
                    constexpr bool nxdlperwave_forward_sweep =
                        (mxdlperwave % (2 * CShuffleMRepeatPerShuffle) == 0);

                    constexpr index_t nxdlperwave_value =
                        nxdlperwave_forward_sweep
                            ? nxdlperwave_iter
                            : (NXdlPerWave - nxdlperwave_iter - CShuffleNRepeatPerShuffle);

                    constexpr auto nxdlperwave = Number<nxdlperwave_value>{};

                    // make sure it's safe to do ds_write
                    block_sync_lds();

                    // VGPR to LDS
                    c_thread_copy_vgpr_to_lds.Run(
                        c_m0_n0_m1_n1_m2_m3_m4_n2_thread_desc,
                        make_tuple(mxdlperwave, nxdlperwave, I0, I0, I0, I0, I0, I0),
                        c_thread_buf,
                        c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                        c_block_buf);

                    // make sure it's safe to do ds_read
                    block_sync_lds();

                    // LDS to global
                    c_block_copy_lds_to_global.Run(c_block_desc_mblock_mperblock_nblock_nperblock,
                                                   c_block_buf,
                                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                   c_grid_buf);

                    // move on nxdlperwave dimension
                    if constexpr(nxdlperwave_forward_sweep &&
                                 (nxdlperwave < NXdlPerWave - CShuffleNRepeatPerShuffle))
                    {
                        c_block_copy_lds_to_global.MoveDstSliceWindow(
                            c_grid_desc_mblock_mperblock_nblock_nperblock,
                            nxdlperwave_forward_step);
                    }
                    else if constexpr((!nxdlperwave_forward_sweep) && (nxdlperwave > 0))
                    {
                        c_block_copy_lds_to_global.MoveDstSliceWindow(
                            c_grid_desc_mblock_mperblock_nblock_nperblock,
                            nxdlperwave_backward_step);
                    }
                });

                // move on mxdlperwave dimension
                if constexpr(mxdlperwave < MXdlPerWave - CShuffleMRepeatPerShuffle)
                {
                    c_block_copy_lds_to_global.MoveDstSliceWindow(
                        c_grid_desc_mblock_mperblock_nblock_nperblock, mxdlperwave_forward_step);
                }
            });
        }
#else
        // output: register to global memory
        {
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
            constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
            constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_grid =
                m_block_data_idx_on_grid + c_thread_mtx_on_block[I0];

            const index_t n_thread_data_on_grid =
                n_block_data_idx_on_grid + c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_grid_idx =
                m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_grid));

            const auto n_thread_data_on_grid_to_n0_n1_n2_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_grid_idx =
                n_thread_data_on_grid_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_grid));

            auto c_thread_copy =
                ThreadwiseTensorSliceTransfer_v1r3<FloatAcc,
                                                   FloatC,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   decltype(c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   CElementwiseOperation,
                                                   Sequence<M0, N0, I1, I1, M2, I1, M4, I1>,
                                                   CThreadTransferSrcDstAccessOrder,
                                                   CThreadTransferSrcDstVectorDim,
                                                   CThreadTransferDstScalarPerVector,
                                                   CGlobalMemoryDataOperation,
                                                   1,
                                                   true>{
                    c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                    make_multi_index(m_thread_data_on_grid_idx[I0],
                                     n_thread_data_on_grid_idx[I0],
                                     m_thread_data_on_grid_idx[I1],
                                     n_thread_data_on_grid_idx[I1],
                                     m_thread_data_on_grid_idx[I2],
                                     m_thread_data_on_grid_idx[I3],
                                     m_thread_data_on_grid_idx[I4],
                                     n_thread_data_on_grid_idx[I2]),
                    c_element_op};

            c_thread_copy.Run(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                              c_thread_buf,
                              c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                              c_grid_buf);
        }
#endif
    }
};

} // namespace ck

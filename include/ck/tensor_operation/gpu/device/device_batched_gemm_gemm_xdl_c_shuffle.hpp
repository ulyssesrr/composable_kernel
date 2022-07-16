#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/device_utility/device_prop.hpp"
#include "ck/device_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm_c_shuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_gemm_xdl_skip_lds.hpp"

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename QGridDesc_K0_M_K1,
          typename KGridDesc_K0_N_K1,
          typename VGridDesc_K0_N_K1,
          typename RGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename QElementwiseOperation,
          typename KElementwiseOperation,
          typename VElementwiseOperation,
          typename PElementwiseOperation,
          typename RElementwiseOperation,
          typename ComputePtrOffsetOfBatch,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_gemm_c_shuffle_xdl(const FloatAB* __restrict__ p_q_grid,
                                          const FloatAB* __restrict__ p_k_grid,
                                          const FloatAB* __restrict__ p_v_grid,
                                          FloatC* __restrict__ p_o_grid,
                                          const index_t batch_count,
                                          const QGridDesc_K0_M_K1 q_grid_desc_k0_m_k1,
                                          const KGridDesc_K0_N_K1 k_grid_desc_k0_n_k1,
                                          const VGridDesc_K0_N_K1 v_grid_desc_k0_n_k1,
                                          const RGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                                              r_grid_desc_mblock_mperblock_nblock_nperblock,
                                          const QElementwiseOperation q_element_op,
                                          const KElementwiseOperation k_element_op,
                                          const VElementwiseOperation v_element_op,
                                          const PElementwiseOperation p_element_op,
                                          const RElementwiseOperation r_element_op,
                                          const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
                                          const Block2CTileMap block_2_ctile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t q_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetQPtrOffset(g_idx)));
    const long_index_t k_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetKPtrOffset(g_idx)));
    const long_index_t v_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetVPtrOffset(g_idx)));
    const long_index_t o_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetRPtrOffset(g_idx)));

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<HasMainKBlockLoop>(
        p_q_grid + a_batch_offset,
        p_k_grid + k_batch_offset,
        p_v_grid + v_batch_offset,
        ck::Tuple<>{},
        p_o_grid + o_batch_offset,
        p_shared,
        q_element_op,
        k_element_op,
        v_element_op,
        p_element_op,
        r_element_op,
        q_grid_desc_k0_m_k1,
        k_grid_desc_k0_n_k1,
        v_grid_desc_k0_n_k1,
        ck::StaticallyIndexedArray<
            typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
            0>{},
        r_grid_desc_mblock_mperblock_nblock_nperblock,
        block_2_ctile_map);
#else
    ignore = p_q_grid;
    ignore = p_v_grid;
    ignore = p_k_grid;
    ignore = p_o_grid;
    ignore = batch_count;
    ignore = q_grid_desc_k0_m_k1;
    ignore = k_grid_desc_k0_m_k1
    ignore = v_grid_desc_k0_n_k1;
    ignore = r_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = q_element_op;
    ignore = k_element_op;
    ignore = v_element_op;
    ignore = p_element_op;
    ignore = r_element_op;
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif
}

template <typename QLayout,
          typename KLayout,
          typename VLayout,
          typename RLayout,
          typename KDataType,
          typename QDataType,
          typename VDataType,
          typename ODataType,
          typename AccDataType,
          typename KElementwiseOperation,
          typename QElementwiseOperation,
          typename VElementwiseOperation,
          typename PElementwiseOperation,
          typename OlementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t NumPrefetch,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t QK1,
          ck::index_t KK1,
          ck::index_t VK1,
          ck::index_t QKMPerXDL,
          ck::index_t QKNPerXDL,
          ck::index_t QKMXdlPerWave,
          ck::index_t QKNXdlPerWave,
          ck::index_t PVMPerXDL,
          ck::index_t PVNPerXDL,
          ck::index_t PVMXdlPerWave,
          ck::index_t PVNXdlPerWave,
          typename QBlockTransferThreadClusterLengths_K0_M_K1,
          typename QBlockTransferThreadClusterArrangeOrder,
          typename QBlockTransferSrcAccessOrder,
          ck::index_t QBlockTransferSrcVectorDim,
          ck::index_t QBlockTransferSrcScalarPerVector,
          ck::index_t QBlockTransferDstScalarPerVector_K1,
          ck::index_t QBlockLdsAddExtraM,
          typename KBlockTransferThreadClusterLengths_K0_N_K1,
          typename KBlockTransferThreadClusterArrangeOrder,
          typename KBlockTransferSrcAccessOrder,
          ck::index_t KBlockTransferSrcVectorDim,
          ck::index_t KBlockTransferSrcScalarPerVector,
          ck::index_t KBlockTransferDstScalarPerVector_K1,
          ck::index_t KBlockLdsAddExtraN,
          typename VBlockTransferThreadClusterLengths_K0_N_K1,
          typename VBlockTransferThreadClusterArrangeOrder,
          typename VBlockTransferSrcAccessOrder,
          ck::index_t VBlockTransferSrcVectorDim,
          ck::index_t VBlockTransferSrcScalarPerVector,
          ck::index_t VBlockTransferDstScalarPerVector_K1,
          ck::index_t VBlockLdsAddExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceBatchedGemmGemmCShuffleXdl : public DeviceBatchedGemmGemmCShuffle<AElementwiseOperation,
                                                                       BElementwiseOperation,
                                                                       CElementwiseOperation>
{
    using DeviceOp = DeviceBatchedGemmGemmCShuffleXdl;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static auto MakeQGridDescriptor_QK0_M_QK1(index_t M, index_t K, index_t StrideQ)
    {
        // not pad M or K
        assert(K % QK1 == 0);

        const auto QK0 = K / QK1;
        const auto q_grid_desc_k0_m_k1 = [&](){
            return make_naive_tensor_descriptor(make_tuple(QK0, M, QK1),
                                                make_tuple(M * QK1, QK1, I1));
        }

        return q_grid_desc_qk0_m_qk1;
    }

    static auto MakeKGridDescriptor_KK0_N_KK1(index_t N, index_t K, index_t StrideK)
    {
        // not pad M or K
        assert(K % KK1 == 0);

        const auto KK0 = K / KK1;

        const auto k_grid_desc_kk0_n_kk1 = make_naive_tensor_descriptor(make_tuple(KK0, N, KK1),
                                                                        make_tuple(KK1 * N, KK1, I1));

        return k_grid_desc_kk0_n_kk1;
    }

    static auto MakeVGridDescriptor_VK0_N_VK1(index_t N, index_t K, index_t StrideV)
    {
        // not pad M or K
        assert(K % VK1 == 0);

        const auto VK0 = K / VK1;

        const auto v_grid_desc_vk0_n_vk1 = make_naive_tensor_descriptor(make_tuple(VK0, N, VK1),
                                                                        make_tuple(VK1 * N, VK1, I1));

        return v_grid_desc_vk0_n_vk1;
    }

    static auto
    MakeOGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t stride_M, index_t stride_N)
    {
        const auto o_grid_desc_mraw_nraw = [&]() {
            return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                make_tuple(stride_M, stride_N));
        }();

        return o_grid_desc_mraw_nraw;
    }

    
    using QGridDesc_K0_M_K1 = decltype(MakeQGridDescriptor_QK0_M_QK1(1, 1, 1));
    using KGridDesc_K0_N_K1 = decltype(MakeKGridDescriptor_KK0_N_KK1(1, 1, 1));
    using VGridDesc_K0_N_K1 = decltype(MakeVGridDescriptor_VK0_N_VK1(1, 1, 1));
    using OGridDesc_M_N     = decltype(MakeOGridDescriptor_M_N(1, 1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmGemmXdlopsSkipLdsV1<
        BlockSize,
        KDataType, // TODO: distinguish A/B datatype
        AccDataType,
        ODataType,
        InMemoryDataOperationEnum::Set,
        QGridDesc_K0_M_K1,
        KGridDesc_K0_N_K1,
        VGridDesc_K0_N_K1,
        OGridDesc_M_N,
        QElementwiseOperation,
        KElementwiseOperation,
        VElementwiseOperation,
        PElementwiseOperation
        OElementwiseOperation,
        QKMPerBlock,
        QKNPerBlock,
        QKMPerXDL,
        QKNPerXDL,
        PVMPerBlock,
        PVNPerBlock,
        PVMPerXDL,
        PVNPerXDL,
        KPerBlock,
        QK1,
        KK1,
        VK1,
        QKMXdlPerWave,
        QKNXdlPerWave,
        PVMXdlPerWave,
        PVNXdlPerWave,
        KBlockTransferThreadClusterLengths_K0_N_K1,
        KBlockTransferThreadClusterArrangeOrder,
        KBlockTransferSrcAccessOrder,
        KBlockTransferSrcVectorDim,
        KBlockTransferSrcScalarPerVector,
        KBlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        KBlockLdsAddExtraM,
        VBlockTransferThreadClusterLengths_K0_N_K1,
        VBlockTransferThreadClusterArrangeOrder,
        VBlockTransferSrcAccessOrder,
        VBlockTransferSrcVectorDim,
        VBlockTransferSrcScalarPerVector,
        VBlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        VBlockLdsAddExtraM,
        QBlockTransferSrcScalarPerVector,
        false,                            // BThreadTransferSrcResetCoordinateAfterRun,
        Sequence<0, 2, 4, 5, 6, 1, 3, 7>, // CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const QDataType* p_q_grid,
                 const KDataType* p_k_grid,
                 const VDataType* p_v_grid,
                 ODataType* p_o_grid,
                 index_t QKM,
                 index_t QKN,
                 index_t PVM,
                 index_t PVN,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t M01,
                 index_t N01,
                 QElementwiseOperation q_element_op,
                 KElementwiseOperation k_element_op,
                 VElementwiseOperation v_element_op,
                 PElementwiseOperation p_element_op,
                 OElementwiseOperation o_element_op)
            : p_q_grid_{p_q_grid},
              p_k_grid_{p_k_grid},
              p_v_grid_{p_v_grid},
              p_o_grid_{p_o_grid}
              q_grid_desc_k0_m_k1_{},
              k_grid_desc_k0_n_k1_{},
              o_grid_desc_m_n_{},
              c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              q_element_op_{q_element_op},
              k_element_op_{k_element_op},
              p_element_op_{p_element_op},
              v_element_op_{v_element_op},
              o_element_op_{o_element_op},
        {
            q_grid_desc_k0_m_k1_ =
                DeviceOp::MakeQGridDescriptor_QK0_M_QK1(QKM, K, StrideA);
            k_grid_desc_k0_n_k1_ =
                DeviceOp::MakeKGridDescriptor_KK0_N_KK1(QKN, K, StrideB);
            v_grid_desc_k0_n_k1_ =
                DeviceOp::MakeVGridDescriptor_VK0_N_VK1(PVN, K, StrideB);
            o_grid_desc_m_n_ = DeviceOp::MakeOGridDescriptor_M_N(PVM, PVN, StrideC);

            if(GridwiseGemm::CheckValidity(
                   q_grid_desc_k0_m_k1_, k_grid_desc_k0_n_k1_, o_grid_desc_m_n_, M01_, N01_))
            {
                c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(o_grid_desc_m_n_);

                block_2_ctile_map_ =
                    GridwiseGemm::MakeDefaultBlock2CTileMap(o_grid_desc_m_n_, M01, N01);

                b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3_ =
                    GridwiseGemm::MakeBGridDescriptor_K0_K1_K2_N0_N1_N2_N3_K3(k_grid_desc_k0_n_k1_);
            }
        }

        //  private:
        const QDataType* p_q_grid_;
        const KDataType* p_k_grid_;
        const VDataType* p_v_grid_;
        ODataType* p_o_grid_;
        QGridDesc_K0_M_K1 q_grid_desc_k0_m_k1_;
        KGridDesc_K0_N_K1 k_grid_desc_k0_n_k1_;
        VGridDesc_K0_N_K1 v_grid_desc_k0_n_k1_;
        OGridDesc_M_N o_grid_desc_m_n_;
        typename GridwiseGemm::BGridDesc_K0_K1_K2_N0_N1_N2_N3_K3
            b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3_;
        typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2
            c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;
        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        QElementwiseOperation q_element_op_;
        KElementwiseOperation k_element_op_;
        PElementwiseOperation p_element_op_;
        VElementwiseOperation v_element_op_;
        OElementwiseOperation o_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            {
                std::cout << "arg.q_grid_desc_k0_m_k1_{" << arg.q_grid_desc_k0_m_k1_.GetLength(I0)
                          << ", " << arg.q_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.q_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.k_grid_desc_k0_n_k1_{" << arg.k_grid_desc_k0_n_k1_.GetLength(I0)
                          << ", " << arg.k_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.k_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.o_grid_desc_m_n_{ " << arg.o_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.o_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }

            if(!GridwiseGemm::CheckValidity(arg.q_grid_desc_k0_m_k1_,
                                            arg.k_grid_desc_k0_n_k1_,
                                            arg.o_grid_desc_m_n_,
                                            arg.M01_,
                                            arg.N01_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3 has invalid setting");
            }

            const index_t grid_size = GridwiseGemm::CalculateGridSize(arg.o_grid_desc_m_n_);

            const auto K0 = arg.q_grid_desc_k0_m_k1_.GetLength(I0);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;

            if(has_main_k0_block_loop)
            {
                const auto kernel = kernel_gemm_xdlops_skip_b_lds_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<typename GridwiseGemm::BGridDesc_K0_K1_K2_N0_N1_N2_N3_K3>,
                    remove_reference_t<typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                    true>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.q_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.a_element_op_,
                                                  arg.b_element_op_,
                                                  arg.c_element_op_,
                                                  arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_gemm_xdlops_skip_b_lds_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<typename GridwiseGemm::BGridDesc_K0_K1_K2_N0_N1_N2_N3_K3>,
                    remove_reference_t<typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                    false>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.q_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_k1_k2_n0_n1_n2_n3_k3_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.a_element_op_,
                                                  arg.b_element_op_,
                                                  arg.c_element_op_,
                                                  arg.block_2_ctile_map_);
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        return GridwiseGemm::CheckValidity(arg.q_grid_desc_k0_m_k1_,
                                           arg.k_grid_desc_k0_n_k1_,
                                           arg.o_grid_desc_m_n_,
                                           arg.M01_,
                                           arg.N01_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
                                                      index_t /* KBatch */ = 1) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          1,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceOp"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << K1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave
            << ">";
        // clang-format on

        return str.str();
    }

};
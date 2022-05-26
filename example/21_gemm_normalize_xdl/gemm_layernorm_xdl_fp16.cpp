#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>

#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_5ary_elementwise_xdl_cshuffle.hpp"
#include "device_gemm_reduce_xdl_cshuffle.hpp"
#include "element_wise_operation.hpp"
#include "reference_gemm.hpp"
#include "gemm_specialization.hpp"
#include "element_wise_reduce_operation.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType                = F16;
using BDataType                = F16;
using CDataType                = F16;
using ReduceAccDataType        = F32;
using DDataType                = F32;
using DPtrsGlobal              = ck::Tuple<DDataType*, DDataType*>;
using GammaDataType            = F16;
using BetaDataType             = F16;
using LayerNormOutDataType     = F16;
using NormalizeComputeDataType = F32;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp  = ck::tensor_operation::element_wise::PassThrough;
using BElementOp  = ck::tensor_operation::element_wise::PassThrough;
using CElementOp  = ck::tensor_operation::element_wise::PassThrough;
using D0ReduceOp  = ck::reduce::Add<ReduceAccDataType>;
using D1ReduceOp  = ck::reduce::Add<ReduceAccDataType>;
using DxsReduceOp = ck::Tuple<D0ReduceOp, D1ReduceOp>;

using UnaryIdenticElementOp =
    ck::tensor_operation::element_wise::UnaryIdentic<ReduceAccDataType, ReduceAccDataType, false>;
using UnaryDivElementOp =
    ck::tensor_operation::element_wise::UnaryIdentic<ReduceAccDataType, ReduceAccDataType, true>;
using UnarySquareElementOp =
    ck::tensor_operation::element_wise::UnarySquare<ReduceAccDataType, ReduceAccDataType, false>;
using DxsInElementOp  = ck::Tuple<UnaryIdenticElementOp, UnarySquareElementOp>;
using DxsOutElementOp = ck::Tuple<UnaryDivElementOp, UnaryDivElementOp>;

using DGlobalMemOp =
    ck::InMemoryDataOperationEnumSequence<ck::InMemoryDataOperationEnum::AtomicAdd,
                                          ck::InMemoryDataOperationEnum::AtomicAdd>;

static constexpr auto GemmSpecialization =
    ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmReduceInstance = ck::tensor_operation::device::DeviceGemmReduce_Xdl_CShuffle
//######| ALayout| BLayout| CLayout|AData| BData| CData|  GemmAcc| CShuffle| ReduceAcc|         DData|           A|           B|           C|         Dxs|     DxsInEleOp|     DxsAccEleOp|             D|               GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|              CReduce| CReduceThreadLds2VGprCopy| CReduceThreadVgpr2GlobalCopy|
//######|        |        |        | Type|  Type|  Type| DataType| DataType|  DataType|    Type Tuple| Elementwise| Elementwise| Elementwise|      Reduce|               |                |    MemoryData|     Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN| MXdlPerWave| NXdlPerWave|            _MBlock_MPerBlock| ScalarPerVector| ThreadClusterLengths|     SrcDstScalarPerVector|        SrcDstScalarPerVector|
//######|        |        |        |     |      |      |         |         |          |              |   Operation|   Operation|   Operation|   Operation|               |                |     Operation|                   |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|            _NBlock_NPerBlock|      _NPerBlock| _MPerBlock_NPerBlock|                _NPerBlock|                   _MPerBlock|
//######|        |        |        |     |      |      |         |         |          |              |            |            |            |            |               |                |              |                   |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |                     |                          |                             |
        <     Row,     Col,     Row,  F16,   F16,   F16,      F32,      F32,       F32,   DPtrsGlobal,  AElementOp,  BElementOp,  CElementOp, DxsReduceOp, DxsInElementOp, DxsOutElementOp,  DGlobalMemOp, GemmSpecialization,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8,             S<64, 4>,                         4,                            1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AElementOp, BElementOp, CElementOp>;

using NormalizeFunctor = ck::tensor_operation::element_wise::Normalize;

// A:x, B:E[x], C:E[x^2], D:Gamma, E:Beta , F:y
using DeviceNormalizeInstance = ck::tensor_operation::device::Device5AryElementwise_Xdl_CShuffle<
    CDataType,
    DDataType,
    DDataType,
    GammaDataType,
    BetaDataType,
    LayerNormOutDataType,
    NormalizeComputeDataType,
    NormalizeFunctor,
    2,
    8,
    8,  // scalarPerVector: gemm_out
    1,  // scalarPerVector: reduce_mean
    1,  // scalarPerVector: reduce_mean_square
    8,  // scalarPerVector: Gamma
    8,  // scalarPerVector: Beta
    8>; // scalarPerVector: LayerNorm_out

int main()
{
    bool time_kernel = false;

    // GEMM shape
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA = 1024;
    ck::index_t StrideB = 1024;
    ck::index_t StrideC = 1024;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor(std::vector<std::size_t>({len}),
                                    std::vector<std::size_t>({stride}));
    };

    auto f_host_tensor_descriptor2d =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor2d(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor2d(K, N, StrideB, BLayout{}));
    Tensor<CDataType> c_m_n(f_host_tensor_descriptor2d(M, N, StrideC, CLayout{}));
    Tensor<DDataType> reduceMean_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<DDataType> reduceMeanSquare_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<GammaDataType> gamma_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<BetaDataType> beta_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<LayerNormOutDataType> layerNorm_m_n(
        f_host_tensor_descriptor2d(M, N, StrideC, CLayout{}));

    a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-5, 5});
    b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-5, 5});
    gamma_n.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{-0.5, 0.5});
    beta_n.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{-0.5, 0.5});

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n.mDesc.GetElementSpace());
    DeviceMem reduceMean_device_buf(sizeof(DDataType) * reduceMean_m.mDesc.GetElementSpace());
    DeviceMem reduceMeanSquare_device_buf(sizeof(DDataType) *
                                          reduceMeanSquare_m.mDesc.GetElementSpace());
    DeviceMem gamma_device_buf(sizeof(GammaDataType) * gamma_n.mDesc.GetElementSpace());
    DeviceMem beta_device_buf(sizeof(BetaDataType) * beta_n.mDesc.GetElementSpace());
    DeviceMem layerNorm_device_buf(sizeof(LayerNormOutDataType) *
                                   layerNorm_m_n.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    gamma_device_buf.ToDevice(gamma_n.mData.data());
    beta_device_buf.ToDevice(beta_n.mData.data());

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};
    auto dxs_global =
        ck::make_tuple(static_cast<DDataType*>(reduceMean_device_buf.GetDeviceBuffer()),
                       static_cast<DDataType*>(reduceMeanSquare_device_buf.GetDeviceBuffer()));

    auto dxs_in_element_op  = DxsInElementOp{};
    auto dxs_out_element_op = DxsOutElementOp{M, M};

    // Prepare GEMM, reduce_mean, reduce_mean_square
    auto gemmReduce         = DeviceGemmReduceInstance{};
    auto gemmReduce_invoker = gemmReduce.MakeInvoker();
    auto gemmReduce_argument =
        gemmReduce.MakeArgument(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                dxs_global,
                                M,
                                N,
                                K,
                                StrideA,
                                StrideB,
                                StrideC,
                                a_element_op,
                                b_element_op,
                                c_element_op,
                                dxs_in_element_op,
                                dxs_out_element_op);

    if(!gemmReduce.IsSupportedArgument(gemmReduce_argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    reduceMean_device_buf.SetZero();
    reduceMeanSquare_device_buf.SetZero();

    // Prepare LayerNorm
    auto layerNorm             = DeviceNormalizeInstance{};
    auto layerNorm_invoker_ptr = layerNorm.MakeInvokerPointer();
    auto layerNorm_argument =
        layerNorm.MakeArgumentPointer(c_device_buf.GetDeviceBuffer(),
                                      reduceMean_device_buf.GetDeviceBuffer(),
                                      reduceMeanSquare_device_buf.GetDeviceBuffer(),
                                      gamma_device_buf.GetDeviceBuffer(),
                                      beta_device_buf.GetDeviceBuffer(),
                                      layerNorm_device_buf.GetDeviceBuffer(),
                                      {M, N},
                                      {StrideC, 1},
                                      {1, 0},
                                      {1, 0},
                                      {0, 1},
                                      {0, 1},
                                      {StrideC, 1},
                                      NormalizeFunctor{});

    if(!layerNorm.IsSupportedArgument(layerNorm_argument.get()))
    {
        throw std::runtime_error("The runtime parameters seems not supported by the "
                                 "Device5AryElementwise_Xdl_CShuffle instance, exiting!");
    }

    // run kernel
    gemmReduce_invoker.Run(gemmReduce_argument, StreamConfig{nullptr, time_kernel});
    layerNorm_invoker_ptr->Run(layerNorm_argument.get(), StreamConfig{nullptr, time_kernel});

    bool pass = true;
    return pass ? 0 : 1;
}

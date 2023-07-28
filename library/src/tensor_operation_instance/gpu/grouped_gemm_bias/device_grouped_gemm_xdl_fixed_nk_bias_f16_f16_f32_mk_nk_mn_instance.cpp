// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_fixed_nk.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using D0DataType = F32;
using DsDataType = ck::Tuple<D0DataType>;

using D0Layout = Row;
using DsLayout = ck::Tuple<D0Layout>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::AddBias;

static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_nk_mn_irregular_tile_instances =
    std::tuple<
        // clang-format off
        //############################|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|      DsData| EData|           A|           B|         CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|        Type|  Type| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|       |       |            |       |      |      |        |         |            |      |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,   256,   128,   256,    32,   8,   8,   32,   32,    2,    4,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,              4>,
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,              4>,
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,   256,   128,    64,    32,   8,   8,   32,   32,    2,    1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,              4>,
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,   256,    64,   128,    32,   8,   8,   32,   32,    1,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,              4>,
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,   128,   128,   128,    32,   8,   8,   32,   32,    4,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 16, 1, 8>,              4>,
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,   128,   128,    64,    32,   8,   8,   32,   32,    2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,              4>,    
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,   128,    64,   128,    32,   8,   8,   32,   32,    2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 16, 1, 8>,              4>,
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,   128,   128,    32,    32,   8,   8,   32,   32,    2,    1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,              4>,
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,   128,    32,   128,    32,   8,   8,   32,   32,    1,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 16, 1, 8>,              4>,
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,   128,    32,   256,    32,   8,   8,   32,   32,    1,    4,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 16, 1, 8>,              4>,
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,    64,    64,    64,    32,   8,   8,   32,   32,    2,    2,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,              4>,
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,    64,    64,    32,    32,   8,   8,   32,   32,    2,    1,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,              4>,
        DeviceGroupedGemm_Xdl_Fixed_NK<    Row,    Col,    DsLayout,    Row,   F16,   F16,     F32,      F32,  DsDataType,   F32, PassThrough, PassThrough,         Add, GemmMNKPadding,        1,    64,    32,    64,    32,   8,   8,   32,   32,    1,    2,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,         1,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,              4>
        // clang-format on
        >;

void add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Col,
                                                         DsLayout,
                                                         Row,
                                                         F16,
                                                         F16,
                                                         DsDataType,
                                                         F32,
                                                         PassThrough,
                                                         PassThrough,
                                                         Add>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_nk_mn_irregular_tile_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

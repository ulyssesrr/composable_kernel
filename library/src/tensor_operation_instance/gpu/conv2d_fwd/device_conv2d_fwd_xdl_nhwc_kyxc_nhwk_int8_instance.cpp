#include <stdlib.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv2d_fwd_instance {

using F32 = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto ConvFwd1x1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0;

static constexpr auto ConvFwd1x1S1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;

// Compilation parameters for in[n, hi, wi, c] * wei[k, y, x, c] = out[n, ho, wo, k]
using device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances = std::tuple<
    // clang-format off
        //################################################################| InData|  WeiData| OutData|  AccData|          In|         Wei|         Out|    ConvForward| Block|  MPer|  NPer| K0Per|  K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|
        //################################################################|   Type|     Type|    Type|     Type| Elementwise| Elementwise| Elementwise| Specialization|  Size| Block| Block| Block|    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|
        //################################################################|       |         |        |         |   Operation|   Operation|   Operation|               |      |      |      |      |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|
        //################################################################|       |         |        |         |            |            |            |               |      |      |      |      |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,   256,   256,   128,     4,  16,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,   256,   128,   256,     4,  16,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,   128,   128,   128,     4,  16,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,   256,   128,   128,     4,  16,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,   128,   128,    64,     4,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,   128,    64,   128,     4,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,    64,    64,    64,     4,  16,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,   256,   128,    64,     4,  16,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,   256,    64,   128,     4,  16,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,   128,   128,    32,     4,  16,   32,   32,    2,    1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,   128,    32,   128,     4,  16,   32,   32,    1,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,    64,    64,    32,     4,  16,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwdDefault,    64,    32,    64,     4,  16,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>
    // clang-format on
    >;

using device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_1x1_p0_int8_instances = std::tuple<
    // clang-format off
        //################################################################| InData|  WeiData| OutData|  AccData|          In|         Wei|         Out|    ConvForward| Block|  MPer|  NPer| K0Per|  K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|
        //################################################################|   Type|     Type|    Type|     Type| Elementwise| Elementwise| Elementwise| Specialization|  Size| Block| Block| Block|    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|
        //################################################################|       |         |        |         |   Operation|   Operation|   Operation|               |      |      |      |      |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|
        //################################################################|       |         |        |         |            |            |            |               |      |      |      |      |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,   256,   256,   128,     4,  16,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,   256,   128,   256,     4,  16,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,   128,   128,   128,     4,  16,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,   256,   128,   128,     4,  16,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,   128,   128,    64,     4,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,   128,    64,   128,     4,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,    64,    64,    64,     4,  16,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,   256,   128,    64,     4,  16,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,   256,    64,   128,     4,  16,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,   128,   128,    32,     4,  16,   32,   32,    2,    1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,   128,    32,   128,     4,  16,   32,   32,    1,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,    64,    64,    32,     4,  16,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough,   ConvFwd1x1P0,    64,    32,    64,     4,  16,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>
    // clang-format on
    >;

using device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_1x1_s1_p0_int8_instances = std::tuple<
    // clang-format off
        //################################################################| InData|  WeiData| OutData|  AccData|          In|         Wei|         Out|    ConvForward| Block|  MPer|  NPer| K0Per|  K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|
        //################################################################|   Type|     Type|    Type|     Type| Elementwise| Elementwise| Elementwise| Specialization|  Size| Block| Block| Block|    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|
        //################################################################|       |         |        |         |   Operation|   Operation|   Operation|               |      |      |      |      |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|
        //################################################################|       |         |        |         |            |            |            |               |      |      |      |      |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,   256,   256,   128,     4,  16,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,   256,   128,   256,     4,  16,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,   128,   128,   128,     4,  16,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,   256,   128,   128,     4,  16,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,   128,   128,    64,     4,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,   128,    64,   128,     4,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,    64,    64,    64,     4,  16,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,   256,   128,    64,     4,  16,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,   256,    64,   128,     4,  16,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,   128,   128,    32,     4,  16,   32,   32,    2,    1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,   128,    32,   128,     4,  16,   32,   32,    1,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,    64,    64,    32,     4,  16,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>,
        DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<  int8_t,  int8_t,  int8_t,  int32_t, PassThrough, PassThrough, PassThrough, ConvFwd1x1S1P0,    64,    32,    64,     4,  16,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,      true,               7,               1>
    // clang-format on
    >;

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(
    std::vector<DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>>& instances)
{
    add_device_operation_instances(instances,
                                   device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances{});
    add_device_operation_instances(instances,
                                   device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_1x1_p0_int8_instances{});
    add_device_operation_instances(instances,
                                   device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_1x1_s1_p0_int8_instances{});
}

} // namespace device_conv2d_fwd_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

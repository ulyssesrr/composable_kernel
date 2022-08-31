// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_bias_gelu_gemm_bias.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

using D0ElementOp = ck::tensor_operation::element_wise::AddRelu;
using D1ElementOp = ck::tensor_operation::element_wise::Add;

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_batched_gemm_bias_gelu_gemm_bias_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmBiasGeluGemmBias<Row,
                                                                  Col,
                                                                  Row,
                                                                  Row,
                                                                  Row,
                                                                  ck::Tuple<Row>,
                                                                  F16,
                                                                  F16,
                                                                  F16,
                                                                  F16,
                                                                  F16,
                                                                  ck::Tuple<F16>,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  D0ElementOp,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  D1ElementOp>>>& instances);

template <typename A0Layout,
          typename B0Layout,
          typename D0Layout,
          typename B1Layout,
          typename C1Layout,
          typename D1sLayout,
          typename A0DataType,
          typename B0DataType,
          typename D0DataType,
          typename B1DataType,
          typename C1DataType,
          typename D1sDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceBatchedGemmBiasGeluGemmBias<A0Layout,
                                                                    B0Layout,
                                                                    D0Layout,
                                                                    B1Layout,
                                                                    C1Layout,
                                                                    D1sLayout,
                                                                    A0DataType,
                                                                    B0DataType,
                                                                    D0DataType,
                                                                    B1DataType,
                                                                    C1DataType,
                                                                    D1sDataType,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    D0ElementOp,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    D1ElementOp>>
{
    using DeviceOp = DeviceBatchedGemmBiasGeluGemmBias<A0Layout,
                                                       B0Layout,
                                                       D0Layout,
                                                       B1Layout,
                                                       C1Layout,
                                                       D1sLayout,
                                                       A0DataType,
                                                       B0DataType,
                                                       D0DataType,
                                                       B1DataType,
                                                       C1DataType,
                                                       D1sDataType,
                                                       PassThrough,
                                                       PassThrough,
                                                       PassThrough,
                                                       D0ElementOp,
                                                       PassThrough,
                                                       PassThrough,
                                                       D1ElementOp>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<A0DataType, half_t> && is_same_v<B0DataType, half_t> &&
                     is_same_v<B1DataType, half_t> && is_same_v<C1DataType, half_t>)
        {
            if constexpr(is_same_v<A0Layout, Row> && is_same_v<B0Layout, Col> &&
                         is_same_v<B1Layout, Row> && is_same_v<C1Layout, Row>)
            {
                add_device_batched_gemm_bias_gelu_gemm_bias_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
                    op_ptrs);
            }
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

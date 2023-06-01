// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include "profiler/profile_batched_gemm_impl.hpp"

#include "ck/library/tensor_operation_instance/gpu/batched_gemm.hpp"

namespace {
using ADataType = ck::half_t;
using BDataType = ck::half_t;
using CDataType = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
} // namespace

int main()
{
    int M          = 512;
    int N          = 256;
    int K          = 128;
    int BatchCount = 3;

    bool pass = true;

    using namespace ck::tensor_operation::device;

    pass = pass && ck::profiler::profile_batched_gemm_impl<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           Row,
                                                           Row,
                                                           Row,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough,
                                                           DeviceBatchedGemm<ALayout,
                                                                             BLayout,
                                                                             CLayout,
                                                                             ADataType,
                                                                             BDataType,
                                                                             CDataType,
                                                                             PassThrough,
                                                                             PassThrough,
                                                                             PassThrough>>(
                       true, 1, false, 1, M, N, K, K, N, N, M * K, K * N, M * N, BatchCount);

    pass = pass && ck::profiler::profile_batched_gemm_impl<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           Row,
                                                           Col,
                                                           Row,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough,
                                                           DeviceBatchedGemm<ALayout,
                                                                             BLayout,
                                                                             CLayout,
                                                                             ADataType,
                                                                             BDataType,
                                                                             CDataType,
                                                                             PassThrough,
                                                                             PassThrough,
                                                                             PassThrough>>(
                       true, 1, false, 1, M, N, K, K, K, N, M * K, K * N, M * N, BatchCount);

    pass = pass && ck::profiler::profile_batched_gemm_impl<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           Col,
                                                           Row,
                                                           Row,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough,
                                                           DeviceBatchedGemm<ALayout,
                                                                             BLayout,
                                                                             CLayout,
                                                                             ADataType,
                                                                             BDataType,
                                                                             CDataType,
                                                                             PassThrough,
                                                                             PassThrough,
                                                                             PassThrough>>(
                       true, 1, false, 1, M, N, K, M, N, N, M * K, K * N, M * N, BatchCount);

    pass = pass && ck::profiler::profile_batched_gemm_impl<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           Col,
                                                           Col,
                                                           Row,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough,
                                                           DeviceBatchedGemm<ALayout,
                                                                             BLayout,
                                                                             CLayout,
                                                                             ADataType,
                                                                             BDataType,
                                                                             CDataType,
                                                                             PassThrough,
                                                                             PassThrough,
                                                                             PassThrough>>(
                       true, 1, false, 1, M, N, K, M, K, N, M * K, K * N, M * N, BatchCount);

    std::cout << "test BatchedGEMM fp16: " << (pass ? "Pass" : "Fail") << std::endl;
    return pass ? 0 : 1;
}

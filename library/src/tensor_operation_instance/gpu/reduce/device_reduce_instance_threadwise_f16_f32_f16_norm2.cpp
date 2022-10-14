// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_threadwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOpId | PropagateNan | UseIndex 
template void add_device_reduce_instance_threadwise<half_t, float, half_t, 4, 3, ReduceTensorOp::NORM2, false, false>(std::vector<deviceReduceThreadWisePtrType<4, 3, ReduceTensorOp::NORM2>>&); 
template void add_device_reduce_instance_threadwise<half_t, float, half_t, 4, 4, ReduceTensorOp::NORM2, false, false>(std::vector<deviceReduceThreadWisePtrType<4, 4, ReduceTensorOp::NORM2>>&); 
template void add_device_reduce_instance_threadwise<half_t, float, half_t, 4, 1, ReduceTensorOp::NORM2, false, false>(std::vector<deviceReduceThreadWisePtrType<4, 1, ReduceTensorOp::NORM2>>&); 
template void add_device_reduce_instance_threadwise<half_t, float, half_t, 2, 1, ReduceTensorOp::NORM2, false, false>(std::vector<deviceReduceThreadWisePtrType<2, 1, ReduceTensorOp::NORM2>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

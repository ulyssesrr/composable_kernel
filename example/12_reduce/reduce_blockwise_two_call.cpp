// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <sstream>

#include <getopt.h>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_reduction.hpp"
#include "ck/library/utility/ranges.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

using InOutDataType = ck::half_t;
using InOutDataType = ck::half_t;
using AccDataType   = float;

constexpr ReduceTensorOp ReduceOpId = ReduceTensorOp::NORM2;
constexpr bool PropagateNan         = true;
constexpr bool OutputIndex          = false;

using ReduceOperation = typename reduce_binary_operator<ReduceOpId>::opType;
using InElementwiseOperation =
    typename reduce_unary_operator<ReduceOpId, true, true>::InElementwiseOperation;
using AccElementwiseOperation =
    typename reduce_unary_operator<ReduceOpId, true, true>::AccElementwiseOperation;

using PassThroughOp = tensor_operation::element_wise::PassThrough;

using DeviceReduceInstance_1 = DeviceReduceMultiBlock<InOutDataType,
                                                      AccDataType,
                                                      InOutDataType,
                                                      5, // Rank
                                                      1, // NumReduceDim
                                                      ReduceOperation,
                                                      InElementwiseOperation,
                                                      PassThroughOp,
                                                      InMemoryDataOperationEnum::Set,
                                                      PropagateNan,
                                                      OutputIndex,
                                                      false, // HaveIndexInputIfOutputIndex
                                                      256,
                                                      32,
                                                      8,
                                                      1,
                                                      1,
                                                      1, // vector dim
                                                      1,
                                                      1>;

using DeviceReduceInstance_2 = DeviceReduceMultiBlock<InOutDataType,
                                                      AccDataType,
                                                      InOutDataType,
                                                      4, // Rank
                                                      1, // NumReduceDim
                                                      ReduceOperation,
                                                      PassThroughOp,
                                                      AccElementwiseOperation,
                                                      InMemoryDataOperationEnum::Set,
                                                      PropagateNan,
                                                      OutputIndex,
                                                      false, // HaveIndexInputIfOutputIndex
                                                      256,
                                                      128,
                                                      2,
                                                      1,
                                                      1,
                                                      1, // vector dim
                                                      1,
                                                      1>;

static bool do_verify;
static int init_method;
static float alpha;
static float beta;
static bool time_kernel;

int main(int argc, char* argv[])
{
    // used by the device reduction
    const std::vector<int> reduceDims_1    = {4};
    const std::vector<int> invariantDims_1 = {0, 1, 2, 3};

    const std::vector<int> reduceDims_2    = {3};
    const std::vector<int> invariantDims_2 = {0, 1, 2};

    // used by the host reduction
    const std::vector<int> reduceDims    = {3, 4};
    const std::vector<int> invariantDims = {0, 1, 2};

    const std::vector<size_t> inLengths_1 = {64, 320, 80, 4, 128};

    // input lengths of the second reduction, which is also the output lengths of the first
    // reduction
    const std::vector<size_t> inLengths_2 = {64, 320, 80, 4};

    const std::vector<size_t> outLengths = {64, 320, 80};

    if(argc == 1)
    {
        do_verify   = true;
        init_method = 2;
        time_kernel = true;
    }
    else if(argc == 4)
    {
        do_verify   = static_cast<bool>(argv[1]);
        init_method = atoi(argv[2]);
        time_kernel = static_cast<bool>(atoi(argv[3]));
    }
    else
    {
        std::ostringstream ostr;

        ostr << "Wrong parameter! " << std::endl
             << "Usage: " << argv[0] << "[verify 0/1] init_method time_kernel" << std::endl;

        throw std::runtime_error(ostr.str());
    };

    alpha = 1.0f;
    beta  = 0.0f;

    Tensor<InOutDataType> in_1(inLengths_1);

    Tensor<InOutDataType> out_ref(outLengths);
    Tensor<InOutDataType> in_2(inLengths_2); // also the output tensor of the first reduction
    Tensor<InOutDataType> out(outLengths);

    auto inStrides_1 = in_1.GetStrides();
    auto inStrides_2 = in_2.GetStrides();
    auto outStrides  = out.GetStrides();

    size_t invariant_total_length = out.GetElementSize();
    size_t reduce_total_length    = in_1.GetElementSize() / invariant_total_length;

    std::size_t num_thread = 1;

    if(do_verify)
    {
        switch(init_method)
        {
        case 0: break;
        case 1:
            in_1.GenerateTensorValue(GeneratorTensor_1<InOutDataType>{1}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_1<InOutDataType>{1}, num_thread);
            break;
        case 2:
            in_1.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
            break;
        default:
            in_1.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-5.0, 5.0}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-5.0, 5.0},
                                            num_thread);
        }

        if(beta != 0.0f)
        {
            ck::ranges::copy(out_ref, out.begin());
        }
    };

    DeviceMem in_1_dev(in_1.GetMemorySize());
    DeviceMem in_2_dev(in_2.GetMemorySize());
    DeviceMem out_dev(out.GetMemorySize());

    in_1_dev.ToDevice(in_1.data());

    if(beta != 0.0f)
        out_dev.ToDevice(out.data());

    InElementwiseOperation in_elementwise_op;
    AccElementwiseOperation acc_elementwise_op;

    std::tie(in_elementwise_op, acc_elementwise_op) =
        reduce_unary_operator<ReduceOpId, true, true>::GetElementwiseOperator(
            static_cast<int32_t>(reduce_total_length));

    if(do_verify)
    {
        ReductionHost<InOutDataType,
                      AccDataType,
                      InOutDataType,
                      ReduceOperation,
                      InElementwiseOperation,
                      AccElementwiseOperation,
                      5, // Rank
                      2, // NumReduceDim
                      PropagateNan,
                      OutputIndex>
            hostReduce(in_1.GetDesc(), out_ref.GetDesc(), invariantDims, reduceDims);

        hostReduce.Run(alpha,
                       in_1.data(),
                       beta,
                       out_ref.data(),
                       nullptr,
                       in_elementwise_op,
                       acc_elementwise_op);
    };

    using Indices = std::vector<ck::index_t>;

    auto reduce_1 = DeviceReduceInstance_1{};

    auto argument_ptr_1 = reduce_1.MakeArgumentPointer(ck::ranges::to<Indices>(inLengths_1),
                                                       ck::ranges::to<Indices>(inStrides_1),
                                                       ck::ranges::to<Indices>(inLengths_2),
                                                       ck::ranges::to<Indices>(inStrides_2),
                                                       reduceDims_1,
                                                       1.0f,
                                                       0.0f,
                                                       in_1_dev.GetDeviceBuffer(),
                                                       nullptr,
                                                       in_2_dev.GetDeviceBuffer(),
                                                       nullptr,
                                                       in_elementwise_op,
                                                       PassThroughOp{});

    if(!reduce_1.IsSupportedArgument(argument_ptr_1.get()))
    {
        std::cout
            << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
            << std::endl;
    };

    auto invoker_ptr_1 = reduce_1.MakeInvokerPointer();

    auto reduce_2 = DeviceReduceInstance_2{};

    auto argument_ptr_2 = reduce_2.MakeArgumentPointer(ck::ranges::to<Indices>(inLengths_2),
                                                       ck::ranges::to<Indices>(inStrides_2),
                                                       ck::ranges::to<Indices>(outLengths),
                                                       ck::ranges::to<Indices>(outStrides),
                                                       reduceDims_2,
                                                       alpha,
                                                       beta,
                                                       in_2_dev.GetDeviceBuffer(),
                                                       nullptr,
                                                       out_dev.GetDeviceBuffer(),
                                                       nullptr,
                                                       PassThroughOp{},
                                                       acc_elementwise_op);

    if(!reduce_2.IsSupportedArgument(argument_ptr_2.get()))
    {
        std::cout
            << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
            << std::endl;
    };

    auto invoker_ptr_2 = reduce_2.MakeInvokerPointer();

    float avg_time_1 = invoker_ptr_1->Run(argument_ptr_1.get(), StreamConfig{nullptr, time_kernel});
    float avg_time_2 = invoker_ptr_2->Run(argument_ptr_2.get(), StreamConfig{nullptr, time_kernel});

    std::size_t num_bytes = invariant_total_length * reduce_total_length * sizeof(InOutDataType) +
                            invariant_total_length * sizeof(InOutDataType);

    float gb_per_sec = num_bytes / 1.E6 / (avg_time_1 + avg_time_2);

    std::cout << "Perf: " << avg_time_1 + avg_time_2 << " ms, " << gb_per_sec << " GB/s, "
              << reduce_1.GetTypeString() << " => " << reduce_2.GetTypeString() << std::endl;

    bool pass = true;

    if(do_verify)
    {
        out_dev.FromDevice(out.data());
        pass = pass && ck::utils::check_err(out, out_ref);
    };

    return (pass ? 0 : 1);
}

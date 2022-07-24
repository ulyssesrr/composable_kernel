#pragma once
#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct BatchedGemmCPermuteDesc
{
    ck::index_t G0_, G1_, M_, N_;
    ck::index_t stride_G0_, stride_G1_, stride_M_, stride_N_;
};

template <typename ALayout,
          typename BLayout,
          typename DLayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceBatchedGemmCPermute : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_c,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t stride_A,
                        index_t stride_B,
                        std::array<index_t, NumDTensor> stride_Ds,
                        index_t batch_stride_A,
                        index_t batch_stride_B,
                        std::array<index_t, NumDTensor> batch_stride_Ds,
                        BatchedGemmCPermuteDesc batched_gemm_c_permute_desc,
                        index_t BatchCount,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck

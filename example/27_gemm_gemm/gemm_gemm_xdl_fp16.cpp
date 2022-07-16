// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_reduce_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using QDataType         = F16;
using KDataType         = F16;
using PDataType         = F16;
using VDataType         = F16;
using RDataType         = F16;
using GemmAccDataType   = F32;

using QLayout = Row;
using KLayout = Col;
using PLayout = Row;
using VLayout = Row;
using RLayout = Row;

using QElementOp = ck::tensor_operation::element_wise::PassThrough;
using KElementOp = ck::tensor_operation::element_wise::PassThrough;
using PElementOp = ck::tensor_operation::element_wise::PassThrough;
using VElementOp = ck::tensor_operation::element_wise::PassThrough;
using RElementOp = ck::tensor_operation::element_wise::PassThrough;

//static constexpr auto GemmSpecialization =
//    ck::tensor_operation::device::GemmSpecialization::Default;

using ReferenceGemmInstanceQKP = ck::tensor_operation::host::ReferenceBatchedGemm<QDataType,
                                                                        KDataType,
                                                                        PDataType,
                                                                        QElementOp,
                                                                        KElementOp,
                                                                        PElementOp>;

using ReferenceGemmInstancePVR = ck::tensor_operation::host::ReferenceBatchedGemm<PDataType,
                                                                        VDataType,
                                                                        RDataType,
                                                                        PElementOp,
                                                                        VElementOp,
                                                                        RElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t N_ = 1024;
    ck::index_t d_ = 64;

#if 0
    ck::index_t M_QKP = N_;
    ck::index_t N_QKP = N_;
    ck::index_t K_QKP = d_;

    ck::index_t M_PVR = N_;
    ck::index_t N_PVR = d_;
    ck::index_t K_PVR = N_;

    ck::index_t StrideQ = d_;
    ck::index_t StrideK = d_;
    ck::index_t StrideP = N_;
    ck::index_t StrideV = d_;
    ck::index_t StrideR = d_;
#endif
    ck::index_t BatchCount = 8 * 12;

    if(argc == 1)
    {
        // do nothing
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 7)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        N_ = std::stoi(argv[4]);
        d_ = std::stoi(argv[5]);

        BatchCount = std::stoi(argv[6]);
#if 0
        M_QKP = N_;
        N_QKP = N_;
        K_QKP = d_;
        M_PVR = N_;
        N_PVR = d_;
        K_PVR = N_;
        StrideQ = d_;
        StrideK = d_;
        StrideP = N_;
        StrideV = d_;
        StrideR = d_;
#endif
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
        printf("arg4 to 6: S (256x), d(128x), BatchCount(32x)\n");
        exit(0);
    }

    auto f_host_tensor_descriptor = [](std::size_t batch_count,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       auto layout) {
        if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                        std::vector<std::size_t>({row * stride, stride, 1}));
        }
        else
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                        std::vector<std::size_t>({col * stride, 1, stride}));
        }
    };

    Tensor<QDataType> q_g_n_d(f_host_tensor_descriptor(BatchCount, N_, d_, d_, QLayout{}));
    Tensor<KDataType> k_g_d_n(f_host_tensor_descriptor(BatchCount, d_, N_, d_, KLayout{}));
    Tensor<PDataType> p_g_n_n(f_host_tensor_descriptor(BatchCount, N_, N_, N_, PLayout{}));
    Tensor<VDataType> v_g_n_d(f_host_tensor_descriptor(BatchCount, N_, d_, d_, VLayout{}));
    Tensor<RDataType> r_g_n_d_host_result(f_host_tensor_descriptor(BatchCount, N_, d_, d_, RLayout{}));
    Tensor<RDataType> r_g_n_d_device_result(f_host_tensor_descriptor(BatchCount, N_, d_, d_, RLayout{}));

    std::cout << "q_g_n_d: " << q_g_n_d.mDesc << std::endl;
    std::cout << "k_g_d_n: " << k_g_d_n.mDesc << std::endl;
    std::cout << "p_g_n_n: " << p_g_n_n.mDesc << std::endl;
    std::cout << "v_g_n_d: " << v_g_n_d.mDesc << std::endl;
    std::cout << "r_g_n_d: " << r_g_n_d_host_result.mDesc << std::endl;

    std::cout << "time kernel: " << time_kernel << std::endl;

    switch (init_method)
    {
    case 0:
        break;
    case 1:
        q_g_n_d.GenerateTensorValue(GeneratorTensor_2<QDataType>{-5, 5});
        k_g_d_n.GenerateTensorValue(GeneratorTensor_2<KDataType>{-5, 5});
        v_g_n_d.GenerateTensorValue(GeneratorTensor_2<VDataType>{-5, 5});
        break;
    default:
        q_g_n_d.GenerateTensorValue(GeneratorTensor_3<QDataType>{0.0, 1.0});
        k_g_d_n.GenerateTensorValue(GeneratorTensor_3<KDataType>{-0.5, 0.5});
        v_g_n_d.GenerateTensorValue(GeneratorTensor_3<VDataType>{-0.5, 0.5});
        break;
    }

    auto q_element_op                     = QElementOp{};
    auto k_element_op                     = KElementOp{};
    auto v_element_op                     = VElementOp{};
    auto p_element_op                     = PElementOp{};
    auto r_element_op                     = RElementOp{};

    DeviceMem q_device_buf(sizeof(QDataType) * q_g_n_d.mDesc.GetElementSpace());
    DeviceMem k_device_buf(sizeof(KDataType) * k_g_d_n.mDesc.GetElementSpace());
    DeviceMem v_device_buf(sizeof(VDataType) * v_g_n_d.mDesc.GetElementSpace());
    DeviceMem r_device_buf(sizeof(RDataType) *
                                 r_g_n_d_device_result.mDesc.GetElementSpace());

    q_device_buf.ToDevice(q_g_n_d.mData.data());
    k_device_buf.ToDevice(k_g_d_n.mData.data());
    v_device_buf.ToDevice(v_g_n_d.mData.data());

    // bool pass = true;
    if(do_verification)
    {
        auto ref_batched_gemmQKP = ReferenceGemmInstanceQKP{};
        auto ref_invokerQKP      = ref_batched_gemmQKP.MakeInvoker();

        auto ref_argumentQKP = ref_batched_gemmQKP.MakeArgument(
            q_g_n_d, k_g_d_n, p_g_n_n, q_element_op, k_element_op, p_element_op);

        auto ref_batched_gemmPVR = ReferenceGemmInstancePVR{};
        auto ref_invokerPVR      = ref_batched_gemmPVR.MakeInvoker();

        auto ref_argumentPVR = ref_batched_gemmPVR.MakeArgument(
            p_g_n_n, v_g_n_d, r_g_n_d_host_result, p_element_op, v_element_op, r_element_op);

        ref_invokerQKP.Run(ref_argumentQKP);
        ref_invokerPVR.Run(ref_argumentPVR);
    }

}

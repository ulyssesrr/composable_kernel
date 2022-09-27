// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "test_batched_gemm_masking_scale_softmax_gemm_permute_util.hpp"

template <typename Tuple>
class TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16
    : public TestBatchedGemmMaskingScaleSoftmaxGemmPermute<Tuple>
{
};

using Masked = std::true_type;
using NoMask = std::false_type;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using CPermuteNumDims_G_M_O =
    S<2, 1, 1>; // "using CLayout = Row" has been replaced by CPermuteNumDims_G_M_O

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<F16, F16, F16, F16, Row, Col, Row, CPermuteNumDims_G_M_O, NoMask>,
    std::tuple<F16, F16, F16, F16, Row, Col, Row, CPermuteNumDims_G_M_O, Masked>
    >;
// clang-format on

TYPED_TEST_SUITE(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, KernelTypes);

TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, Test_FP16) { this->Run(); }

TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, Test_FP16_PadM)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {136, 128, 32, 128, 2, 3},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, Test_FP16_PadN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 136, 32, 128, 3, 2},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, Test_FP16_PadK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 40, 128, 2, 4},
        {128, 128, 136, 128, 4, 2},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, Test_FP16_PadO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 136, 1, 3},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, Test_FP16_OddM)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {129, 128, 32, 128, 2, 3},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, Test_FP16_OddN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 129, 32, 128, 4, 3},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, Test_FP16_OddK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 33, 128, 2, 3},
        {128, 128, 129, 128, 2, 3},
    };
    this->Run();
}

// If kernel B1Layout is RowMajor, expect not to support odd O size
TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, Test_FP16_OddO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 129, 2, 3},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, DISABLED_Bench_FP16_IrregularK)
{
    this->lengths_ = std::vector<std::vector<int>>{{256, 256, 160, 160, 1, 16},
                                                   {256, 64, 160, 64, 1, 16},
                                                   {1024, 1024, 80, 80, 1, 16},
                                                   {1024, 64, 80, 64, 1, 16},
                                                   {4096, 4096, 40, 40, 1, 16},
                                                   {4096, 64, 40, 64, 1, 16}};
    this->bench_   = true;
    this->verify_  = false;
    this->Run();
}

TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, DISABLED_Bench_FP16)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {256, 256, 64, 64, 48, 16},
        {256, 256, 128, 128, 48, 16},
        {512, 512, 64, 64, 48, 16},
        {512, 512, 128, 128, 48, 16},
        {1024, 1024, 64, 64, 48, 16},
        {1024, 1024, 128, 128, 48, 16},
        {2048, 2048, 64, 64, 48, 16},
        {2048, 2048, 128, 128, 48, 16},
        {4096, 4096, 64, 64, 48, 16},
        {4096, 4096, 128, 128, 48, 16},
    };
    this->bench_  = true;
    this->verify_ = false;
    this->Run();
}

using ck::tensor_operation::device::GemmSpecialization;

TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteInterface, GemmSpecializationSizeMatch)
{
    int P = 120; // requires padding
    int Q = 128; // do not require padding

    // IsSupported(M, N, K, O)
    // clang-format off
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::Default>{}.IsSupported(Q, Q, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MPadding>{}.IsSupported(P, Q, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NPadding>{}.IsSupported(Q, P, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::KPadding>{}.IsSupported(Q, Q, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNPadding>{}.IsSupported(P, P, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MKPadding>{}.IsSupported(P, Q, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NKPadding>{}.IsSupported(Q, P, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKPadding>{}.IsSupported(P, P, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::OPadding>{}.IsSupported(Q, Q, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MOPadding>{}.IsSupported(P, Q, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NOPadding>{}.IsSupported(Q, P, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::KOPadding>{}.IsSupported(Q, Q, P, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNOPadding>{}.IsSupported(P, P, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MKOPadding>{}.IsSupported(P, Q, P, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NKOPadding>{}.IsSupported(Q, P, P, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(P, P, P, P));
    // clang-format on
}

TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteInterface, GemmSpecializationSizeMismatch)
{
    // IsSupported(M, N, K, O)
    // clang-format off
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::Default>{}.IsSupported(128, 128, 120, 128));
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKPadding>{}.IsSupported(128, 128, 128, 120));
    // Kernel can't support odd K size because SrcVectorDim == KDim and must satisfy SizeKRaw % ABSrcScalarPerVector == 0
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(128, 128, 129, 128));
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(128, 128, 130, 128));
    // Kernel can't support odd O size because SrcVectorDim == ODim and must satisfy SizeORaw % B1SrcScalarPerVector == 0
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(128, 128, 128, 129));
    // clang-format on
}

TYPED_TEST(TestBatchedGemmMaskingScaleSoftmaxGemmPermuteFP16, AdhocTest)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {49, 49, 64, 64, 4, 6},
        {64, 49, 64, 64, 4, 6},
        {1020, 1020, 64, 128, 4, 6},
        {576, 576, 64, 64, 4, 6},
    };
    this->bench_ = true;
    this->Run();
}

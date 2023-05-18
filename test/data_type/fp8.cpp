// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"

using ck::f8_t;
using ck::type_convert;
using ck::f8_convert_sr;

TEST(FP8, NumericLimits)
{
    EXPECT_EQ(ck::NumericLimits<f8_t>::Min(), 0x08);
    EXPECT_EQ(ck::NumericLimits<f8_t>::Max(), 0x77);
    EXPECT_EQ(ck::NumericLimits<f8_t>::Lowest(), 0xF7);
    EXPECT_EQ(ck::NumericLimits<f8_t>::QuietNaN(), 0x80);
}

TEST(FP8, ConvertFP32Nearest)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // convert 0 float to fp8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(type_convert<f8_t>(0.0f)), abs_tol);
    // convert minimal float to fp8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(), type_convert<float>(type_convert<f8_t>(std::numeric_limits<float>::min())), abs_tol);
    // convert maximal f8_t to float and check if equal to 240.0
    ASSERT_NEAR(240.0f, type_convert<float>(type_convert<f8_t>(240.0f)), abs_tol);
    // convert maximal float to fp8 and back, check if clipped to 240.0
    ASSERT_NEAR(240.0f, type_convert<float>(type_convert<f8_t>(std::numeric_limits<float>::max())), abs_tol);
    // convert inf float to f8_t and check if it is qNan
    ASSERT_NEAR(0x80, type_convert<f8_t>(std::numeric_limits<float>::infinity()), abs_tol);
    // positive float value to fp8 and back, check if holds
    float pos_float = 0.0078125f;
    ASSERT_NEAR(pos_float, type_convert<float>(type_convert<f8_t>(pos_float)), abs_tol);
    // negative float value to fp8 and back, check if holds
    float neg_float = 0.0156250f;
    ASSERT_NEAR(neg_float, type_convert<float>(type_convert<f8_t>(neg_float)), abs_tol);
}

TEST(FP8, ConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // convert 0 float to fp8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f8_convert_sr<f8_t>(0.0f)), abs_tol);
    // convert minimal float to fp8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(), type_convert<float>(f8_convert_sr<f8_t>(std::numeric_limits<float>::min())), abs_tol);
    // convert maximal f8_t to float and check if equal to 240.0
    ASSERT_NEAR(240.0f, type_convert<float>(f8_convert_sr<f8_t>(240.0f)), abs_tol);
    // convert maximal float to fp8 and back, check if clipped to 240.0
    ASSERT_NEAR(240.0f, type_convert<float>(f8_convert_sr<f8_t>(std::numeric_limits<float>::max())), abs_tol);
    // convert inf float to f8_t and check if it is qNan
    ASSERT_NEAR(0x80, f8_convert_sr<f8_t>(std::numeric_limits<float>::infinity()), abs_tol);
    // positive float value to fp8 and back, check if holds
    float pos_float = 0.0078125f;
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_sr<f8_t>(pos_float)), abs_tol);
    // negative float value to fp8 and back, check if holds
    float neg_float = 0.0156250f;
    ASSERT_NEAR(neg_float, type_convert<float>(f8_convert_sr<f8_t>(neg_float)), abs_tol);
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

// Assume
//  1) XDesc is known at compile-time
//  2) MeanVarDesc is known at compile-time
//  3) XBuffer is static buffer
//  4) MeanBuffer is static buffer
//  5) VarBuffer is static buffer
template <typename AccDataType,
          typename XThreadDesc_M_K,
          typename MeanVarThreadDesc_M,
          bool GetActualVariance = false>
struct ThreadwiseWelford
{
    static constexpr auto x_thread_desc_m_k      = XThreadDesc_M_K{};
    static constexpr auto mean_var_thread_desc_m = MeanVarThreadDesc_M{};

    static constexpr auto thread_x_length_m        = x_thread_desc_m_k.GetLength(Number<0>{});
    static constexpr auto thread_x_length_k        = x_thread_desc_m_k.GetLength(Number<1>{});
    static constexpr auto thread_mean_var_length_m = mean_var_thread_desc_m.GetLength(Number<0>{});

    static_assert(thread_x_length_m == thread_mean_var_length_m,
                  "lengths of source and mean/var buffer must match!");

    static_assert(thread_x_length_k > 0, "lengths of k must greater than 0!");

    __device__ static inline void Update(AccDataType& mean, AccDataType& var, AccDataType x, int K)
    {
        using ck::math::isnan;

        if(isnan(x))
        {
            mean = x;
            var  = x;
        }
        else
        {
            AccDataType delta = x - mean;
            mean += delta / K;
            AccDataType delta2 = x - mean;
            var += delta * delta2;
        }
    }

    template <typename XBufferType, typename MeanBufferType, typename VarBufferType>
    __device__ static void
    Run(const XBufferType& x_buf_m_k, MeanBufferType& mean_buf_m, VarBufferType& var_buf_m, int K)
    {
        mean_buf_m(Number<0>{}) = x_buf_m_k(Number<0>{});
        var_buf_m(Number<0>{})  = 0;

        static_for<1, thread_x_length_m, 1>{}([&](auto iM) {
            constexpr index_t out_offset = mean_var_thread_desc_m.CalculateOffset(make_tuple(iM));

            static_for<1, thread_x_length_k, 1>{}([&](auto iK) {
                constexpr auto in_offset = x_thread_desc_m_k.CalculateOffset(make_tuple(iM, iK));
                Update(mean_buf_m(Number<out_offset>{}),
                       var_buf_m(Number<out_offset>{}),
                       x_buf_m_k[Number<in_offset>{}],
                       K);
            });

            // If we need to merge variance from other thread, we should not get actual variance now
            if constexpr(GetActualVariance)
                var_buf_m(Number<out_offset>{}) = var_buf_m(Number<out_offset>{}) / K;
        });
    };
};

} // namespace ck

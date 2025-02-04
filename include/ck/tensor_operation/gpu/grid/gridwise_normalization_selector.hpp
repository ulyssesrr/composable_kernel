// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/grid/gridwise_normalization_naive_variance.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_normalization_welford_variance.hpp"

namespace ck {
template <typename GridwiseReduction,
          typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename ComputeDataType,
          typename YElementwiseOperation,
          typename GridDesc_M_K>
__global__ void kernel_normalization(const GridDesc_M_K x_grid_desc_m_k,
                                     const GridDesc_M_K gamma_grid_desc_m_k,
                                     const GridDesc_M_K beta_grid_desc_m_k,
                                     const GridDesc_M_K y_grid_desc_m_k,
                                     index_t num_k_block_tile_iteration,
                                     ComputeDataType epsilon,
                                     const XDataType* const __restrict__ p_x_global,
                                     const GammaDataType* const __restrict__ p_gamma_global,
                                     const BetaDataType* const __restrict__ p_beta_global,
                                     YDataType* const __restrict__ p_y_global,
                                     const YElementwiseOperation y_elementwise_op)
{
    GridwiseReduction::Run(x_grid_desc_m_k,
                           gamma_grid_desc_m_k,
                           beta_grid_desc_m_k,
                           y_grid_desc_m_k,
                           num_k_block_tile_iteration,
                           epsilon,
                           p_x_global,
                           p_gamma_global,
                           p_beta_global,
                           p_y_global,
                           y_elementwise_op);
};

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename ComputeDataType,
          typename YElementwiseOperation,
          typename GridDesc_M_K,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XSrcVectorDim,
          index_t XSrcVectorSize,
          index_t GammaSrcVectorDim,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorDim,
          index_t BetaSrcVectorSize,
          index_t YDstVectorDim,
          index_t YDstVectorSize,
          bool UseWelford>
auto NormalizationKernelSelector(bool isSweepOnce)
{
    using GridwiseNormalizationGenericNaive =
        GridwiseNormalizationNaiveVariance_mk_to_mk<XDataType,
                                                    GammaDataType,
                                                    BetaDataType,
                                                    YDataType,
                                                    ComputeDataType,
                                                    YElementwiseOperation,
                                                    GridDesc_M_K,
                                                    BlockSize,
                                                    MThreadClusterSize,
                                                    KThreadClusterSize,
                                                    MThreadSliceSize,
                                                    KThreadSliceSize,
                                                    XSrcVectorDim,
                                                    XSrcVectorSize,
                                                    GammaSrcVectorDim,
                                                    GammaSrcVectorSize,
                                                    BetaSrcVectorDim,
                                                    BetaSrcVectorSize,
                                                    YDstVectorDim,
                                                    YDstVectorSize,
                                                    false>;
    using GridwiseNormalizationSweepOnceNaive =
        GridwiseNormalizationNaiveVariance_mk_to_mk<XDataType,
                                                    GammaDataType,
                                                    BetaDataType,
                                                    YDataType,
                                                    ComputeDataType,
                                                    YElementwiseOperation,
                                                    GridDesc_M_K,
                                                    BlockSize,
                                                    MThreadClusterSize,
                                                    KThreadClusterSize,
                                                    MThreadSliceSize,
                                                    KThreadSliceSize,
                                                    XSrcVectorDim,
                                                    XSrcVectorSize,
                                                    GammaSrcVectorDim,
                                                    GammaSrcVectorSize,
                                                    BetaSrcVectorDim,
                                                    BetaSrcVectorSize,
                                                    YDstVectorDim,
                                                    YDstVectorSize,
                                                    true>;
    using GridwiseNormalizationGenericWelford =
        GridwiseNormalizationWelfordVariance_mk_to_mk<XDataType,
                                                      GammaDataType,
                                                      BetaDataType,
                                                      YDataType,
                                                      ComputeDataType,
                                                      YElementwiseOperation,
                                                      GridDesc_M_K,
                                                      BlockSize,
                                                      MThreadClusterSize,
                                                      KThreadClusterSize,
                                                      MThreadSliceSize,
                                                      KThreadSliceSize,
                                                      XSrcVectorDim,
                                                      XSrcVectorSize,
                                                      GammaSrcVectorDim,
                                                      GammaSrcVectorSize,
                                                      BetaSrcVectorDim,
                                                      BetaSrcVectorSize,
                                                      YDstVectorDim,
                                                      YDstVectorSize,
                                                      false>;
    using GridwiseNormalizationSweepOnceWelford =
        GridwiseNormalizationWelfordVariance_mk_to_mk<XDataType,
                                                      GammaDataType,
                                                      BetaDataType,
                                                      YDataType,
                                                      ComputeDataType,
                                                      YElementwiseOperation,
                                                      GridDesc_M_K,
                                                      BlockSize,
                                                      MThreadClusterSize,
                                                      KThreadClusterSize,
                                                      MThreadSliceSize,
                                                      KThreadSliceSize,
                                                      XSrcVectorDim,
                                                      XSrcVectorSize,
                                                      GammaSrcVectorDim,
                                                      GammaSrcVectorSize,
                                                      BetaSrcVectorDim,
                                                      BetaSrcVectorSize,
                                                      YDstVectorDim,
                                                      YDstVectorSize,
                                                      true>;

    if constexpr(UseWelford)
    {
        return isSweepOnce ? kernel_normalization<GridwiseNormalizationSweepOnceWelford,
                                                  XDataType,
                                                  GammaDataType,
                                                  BetaDataType,
                                                  YDataType,
                                                  ComputeDataType,
                                                  YElementwiseOperation,
                                                  GridDesc_M_K>
                           : kernel_normalization<GridwiseNormalizationGenericWelford,
                                                  XDataType,
                                                  GammaDataType,
                                                  BetaDataType,
                                                  YDataType,
                                                  ComputeDataType,
                                                  YElementwiseOperation,
                                                  GridDesc_M_K>;
    }
    else
    {
        return isSweepOnce ? kernel_normalization<GridwiseNormalizationSweepOnceNaive,
                                                  XDataType,
                                                  GammaDataType,
                                                  BetaDataType,
                                                  YDataType,
                                                  ComputeDataType,
                                                  YElementwiseOperation,
                                                  GridDesc_M_K>
                           : kernel_normalization<GridwiseNormalizationGenericNaive,
                                                  XDataType,
                                                  GammaDataType,
                                                  BetaDataType,
                                                  YDataType,
                                                  ComputeDataType,
                                                  YElementwiseOperation,
                                                  GridDesc_M_K>;
    }
}

} // namespace ck
